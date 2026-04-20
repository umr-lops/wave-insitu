"""
kub.py
======
Loader for Kyoto University SPOT buoys (KUB) NetCDF files.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)
"""

import xarray as xr
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections.abc import Iterable

from ..utils import load_mapping, build_reverse_lookup


# ---------------------------------------------------------------------------
# KUB ID extraction
# ---------------------------------------------------------------------------

def _extract_kub_info(path: Path) -> dict:
    """
    Extract the buoy ID and TC name from the file path.

    The filename stem is the buoy ID, e.g., "SPOT-1269"
    The directory name contains the TC name, e.g., "TCswell_2021_CEMPAKA_minP980" -> "CEMPAKA"
    """
    buoy_id = path.stem.split('-')[1] if '-' in path.stem else path.stem
    dirname = path.parent.name
    # Assuming format: TCswell_YYYY_TCNAME_...
    parts = dirname.split('_')
    if len(parts) >= 3:
        tc_name = parts[2]
    else:
        tc_name = "unknown"
    return {"buoy_id": buoy_id, "tc_name": tc_name}


# ---------------------------------------------------------------------------
# Dataset normalisation
# ---------------------------------------------------------------------------

def normalize_dataset(
    ds: xr.Dataset,
    path: Path,
    reverse_lookup: dict,
) -> pd.DataFrame:
    """
    Normalise a KUB xr.Dataset into a uniform DataFrame.

    Guaranteed columns  : time, latitude, longitude, platform_id, tc_name,
                          provider, platform_type
    Optional columns    : significant_wave_height, dominant_wave_period,
                          wind_speed, wind_direction
    """
    # --- 1. Resolve spatiotemporal coordinate names ---
    all_names = list(ds.coords) + list(ds.data_vars)
    all_names_lower = {nm.lower(): nm for nm in all_names}

    def find_coord(canonical: str) -> str | None:
        for alias, can in reverse_lookup.items():
            if can == canonical and alias in all_names_lower:
                return all_names_lower[alias]
        return None

    time_col = find_coord("time")
    lat_col  = find_coord("latitude")
    lon_col  = find_coord("longitude")

    if not all([time_col, lat_col, lon_col]):
        missing = [
            k for k, v in {
                "time": time_col, "latitude": lat_col, "longitude": lon_col
            }.items() if not v
        ]
        raise ValueError(f"Missing coordinates: {missing}")

    # --- 2. Extract KUB buoy ID and TC name from filename ---
    info = _extract_kub_info(path)
    buoy_id = info["buoy_id"]
    tc_name = info["tc_name"]

    # --- 3. Extract base coordinates (assuming 1D time dimension) ---
    time = ds[time_col].values
    lat = ds[lat_col].values.astype(float)
    lon = ds[lon_col].values.astype(float)

    # --- 4. Build base DataFrame ---
    df = pd.DataFrame({
        "time":         pd.to_datetime(time),
        "latitude":     lat,
        "longitude":    lon,
        "platform_id":  buoy_id,
        "tc_name":      tc_name,
        "provider":     "Kyoto University",
        "platform_type": "spotter",
    })

    # --- 5. Add variables of interest ---
    TARGET_VARIABLES = {
        "significant_wave_height",
        "dominant_wave_period",
        "wind_speed",
        "wind_direction",
    }

    for var_name in ds.data_vars:
        canonical = reverse_lookup.get(var_name.lower())
        if canonical in TARGET_VARIABLES and canonical not in df.columns:
            df[canonical] = ds[var_name].values.astype(float)

    # --- 6. Remove exact duplicate samples (time, position, Hs) ---
    dedupe_subset = ["time", "latitude", "longitude"]
    if "significant_wave_height" in df.columns:
        dedupe_subset.append("significant_wave_height")
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_subset, keep="first")
    dropped = before - len(df)
    # if dropped > 0:
    #     print(f"  {path.name}: removed {dropped} duplicate row(s) with same time, position and Hs.")

    # --- 7. Sort by time ---
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_kub_file(
    path: str | Path,
    mapping: dict,
) -> pd.DataFrame:
    """
    Load a KUB NetCDF file and return a normalised DataFrame.

    Parameters
    ----------
    path : path to the NetCDF file
    mapping : loaded YAML mapping dict
    """
    path = Path(path)
    base_name = path.stem
    dirname = path.parent.name
    reverse_lookup = build_reverse_lookup(mapping)

    ds = xr.open_dataset(path, decode_times=True)
    try:
        df = normalize_dataset(ds, path, reverse_lookup)
    finally:
        ds.close()

    df["source_file"] = str(path)
    df["name"]        = dirname + "_" + base_name

    # Drop NaN coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Drop files where significant_wave_height is entirely NaN
    if "significant_wave_height" in df.columns and not df.empty:
        if df["significant_wave_height"].isna().all():
            print(f"  {base_name}: dropped (Hs all-NaN)")
            return pd.DataFrame()

    return df


def _haversine_distance(lat1, lon1, lat2, lon2) -> pd.Series:
    """Compute great-circle distance in kilometers between two points."""
    lat1 = np.deg2rad(lat1.astype(float))
    lon1 = np.deg2rad(lon1.astype(float))
    lat2 = np.deg2rad(lat2.astype(float))
    lon2 = np.deg2rad(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return 6371.0 * c


def assign_kub_segments(
    df: pd.DataFrame,
    max_time_delta: pd.Timedelta = pd.Timedelta(hours=3),
    max_distance_km: float = 100.0,
) -> pd.DataFrame:
    """Assign a segment index to each buoy track based on time and distance gaps."""
    if df.empty:
        return df

    df = df.sort_values(["platform_id", "time"]).reset_index(drop=True)

    df["time_diff"] = df.groupby("platform_id")["time"].diff()
    df["prev_latitude"] = df.groupby("platform_id")["latitude"].shift(1)
    df["prev_longitude"] = df.groupby("platform_id")["longitude"].shift(1)
    df["distance_km"] = _haversine_distance(
        df["prev_latitude"],
        df["prev_longitude"],
        df["latitude"],
        df["longitude"],
    )

    df["break_segment"] = (
        df["time_diff"].isna()
        | (df["time_diff"] > max_time_delta)
        | (df["distance_km"] > max_distance_km)
    )

    df["segment"] = df.groupby("platform_id")["break_segment"].cumsum().astype(int)
    df["name"] = df["platform_id"].astype(str) + "_" + df["segment"].astype(str)

    # Drop intermediate segmentation columns before returning.
    df = df.drop(columns=["time_diff", "prev_latitude", "prev_longitude", "distance_km", "break_segment", "segment"])

    return df


def load_kub_catalog(
    paths: Iterable[str | Path],
    mapping: dict,
    provider: str | None = None,
    assign_segments: bool = True,
    max_time_delta: pd.Timedelta = pd.Timedelta(hours=3),
    max_distance_km: float = 100.0,
) -> pd.DataFrame:
    """
    Load files from a list or generator of NetCDF paths and return a normalised,
    concatenated DataFrame.

    Parameters
    ----------
    paths : iterable of str or Path
        Paths to NetCDF files, e.g. a list of files or a Path.rglob() generator.
    mapping : dict
        Loaded YAML mapping dict.
    provider : str, optional
        Optional provider label to attach to every loaded DataFrame.
    """
    frames = []
    errors = []

    for path in paths:
        try:
            df = load_kub_file(path, mapping)
            if provider is None:
                provider = "Kyoto University"
            df["provider"] = provider
            frames.append(df)
        except Exception as e:
            errors.append({"file": Path(path), "error": str(e)})

    if errors:
        print(f"Warning: {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"  - {err['file'].name}: {err['error']}")

    if not frames:
        raise RuntimeError("No files could be loaded.")

    df = pd.concat(frames, ignore_index=True)

    # Remove duplicate measurements for the same buoy by time and position.
    df = df.drop_duplicates(subset=["platform_id", "time", "latitude", "longitude"], keep="first")

    if assign_segments:
        df = assign_kub_segments(
            df,
            max_time_delta=max_time_delta,
            max_distance_km=max_distance_km,
        )
    else:
        df = df.sort_values(["platform_id", "time"]).reset_index(drop=True)

    return df