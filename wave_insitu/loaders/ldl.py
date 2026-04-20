"""
ldl_loader.py
=============
Loader for LDL (Lagrangian Drifter Laboratory) Directional Wave Spectral
Drifter (DWSD) NetCDF files.

Each file contains one drifter, one time dimension, and wave/ocean variables
quality-controlled via wave_qcflag1 (0 = good data).

Metadata extracted from the global attribute 'id':
  Format  : LDL_SENSOR_<buoy_id>_<tc_name>.nc
  Example : LDL_SENSOR_300534061905650_MILTON.nc
  -> buoy_id  = "300534061905650"
  -> tc_name  = "MILTON"

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)
"""

import xarray as xr
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from ..utils import load_mapping, build_reverse_lookup


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

# Mapping from dataset variable names to canonical names used in the DataFrame.
# These match the canonical names already used in saildrone_map.py (VARIABLES dict).
LDL_VAR_MAP = {
    "significant_wave_height": "significant_wave_height",
    "peak_period":             "dominant_wave_period",
    "average_period":          "average_period",
    "dominant_direction":      "dominant_wave_direction",
    "sea_surface_temperature": "sea_surface_temperature",
    "sea_level_pressure":      "sea_level_pressure",
}

# QC flag variable applied to all wave variables
QC_FLAG_VAR  = "wave_qcflag1"
QC_GOOD_VALUE = 0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _parse_ldl_id(ds: xr.Dataset, path: Path) -> tuple[str, str, str]:
    """
    Parse the 'id' global attribute to extract buoy_id and tc_name.
    Also returns the full file stem as 'name'.

    Format: LDL_SENSOR_<buoy_id>_<tc_name>.nc
    Positions after split('_'): [0]=LDL, [1]=SENSOR, [2]=buoy_id, [-1]=tc_name

    Falls back to path stem if attribute is missing or malformed.

    Returns
    -------
    name     : file stem without extension  (e.g. "LDL_SENSOR_300534061905650_MILTON")
    buoy_id  : drifter ID string            (e.g. "300534061905650")
    tc_name  : associated tropical cyclone name  (e.g. "MILTON")
    """
    name = path.stem  # default fallback

    raw_id = ds.attrs.get("id", "")
    if not raw_id:
        return name, "unknown", "unknown"

    # Strip extension if present
    stem = raw_id.replace(".nc", "").replace(".NC", "")
    parts = stem.split("_")

    # Expected: LDL_SENSOR_<buoy_id>_<tc_name>
    # buoy_id is at index 2, tc_name is the last part
    try:
        buoy_id = parts[2] if len(parts) > 2 else "unknown"
        tc_name = parts[-1] if len(parts) > 3 else "unknown"
    except IndexError:
        buoy_id = "unknown"
        tc_name = "unknown"

    return name, buoy_id, tc_name


# ---------------------------------------------------------------------------
# QC filtering
# ---------------------------------------------------------------------------

def _apply_qc(ds: xr.Dataset) -> xr.Dataset:
    """
    Mask all data variables where wave_qcflag1 != 0.
    Only the time steps with wave_qcflag1 == 0 are retained.
    Coordinates (time, latitude, longitude) are always preserved.
    """
    if QC_FLAG_VAR not in ds.data_vars:
        raise ValueError(f"QC flag variable '{QC_FLAG_VAR}' not found in dataset")
    
    good_mask = ds[QC_FLAG_VAR] == QC_GOOD_VALUE

    return ds.where(good_mask, drop=True)

    # # Apply mask to wave/ocean variables only (not to coordinates)
    # COORDS_TO_PRESERVE = {"latitude", "longitude", "time", "platform_ID"}
    # vars_to_mask = [
    #     v for v in ds.data_vars
    #     if v not in COORDS_TO_PRESERVE and v != QC_FLAG_VAR
    # ]

    # masked_vars = {}
    # for var in vars_to_mask:
    #     masked_vars[var] = ds[var].where(good_mask)

    # # Build new dataset with masked variables
    # ds_masked = ds.copy()
    # for var, da in masked_vars.items():
    #     ds_masked[var] = da

    # return ds_masked


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def normalize_ldl_dataset(ds: xr.Dataset, path: Path) -> pd.DataFrame:
    """
    Normalise a single LDL DWSD xr.Dataset into a uniform DataFrame.

    Guaranteed columns  : time, latitude, longitude, name, platform_id, tc_name,
                          provider, platform_type
    Optional columns    : significant_wave_height, dominant_wave_period,
                          average_period, dominant_wave_direction,
                          sea_surface_temperature, sea_level_pressure

    QC filtering (wave_qcflag1 == 0) is applied before extraction.
    """
    # --- 1. Parse metadata ---
    name, buoy_id, tc_name = _parse_ldl_id(ds, path)

    # --- 2. Apply QC flag ---
    ds = _apply_qc(ds)

    # --- 3. Extract time and coordinates ---
    if "time" not in ds.coords:
        raise ValueError(f"No 'time' coordinate found in {path.name}")

    time = ds["time"].values
    n    = len(time)

    if "latitude" not in ds.data_vars and "latitude" not in ds.coords:
        raise ValueError(f"No 'latitude' variable found in {path.name}")
    if "longitude" not in ds.data_vars and "longitude" not in ds.coords:
        raise ValueError(f"No 'longitude' variable found in {path.name}")

    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)

    # --- 4. Build base DataFrame ---
    df = pd.DataFrame({
        "time":         pd.to_datetime(time),
        "latitude":     lat,
        "longitude":    lon,
        "name":         name,
        "platform_id":  buoy_id,
        "tc_name":      tc_name,
        "provider":     "LDL (NOPP)",
        "platform_type": "dwsd",
    })

    # --- 5. Add wave / ocean variables using LDL_VAR_MAP ---
    for raw_var, canonical in LDL_VAR_MAP.items():
        if raw_var in ds.data_vars and canonical not in df.columns:
            df[canonical] = ds[raw_var].values.astype(float)

    # --- 6. Sort by time ---
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_ldl_file(path: str | Path) -> pd.DataFrame:
    """
    Load a single LDL DWSD NetCDF file and return a normalised DataFrame.

    Parameters
    ----------
    path : path to the NetCDF file

    Returns
    -------
    DataFrame with columns: time, latitude, longitude, name, platform_id,
    tc_name, provider, platform_type, source_file + available wave/ocean variables.
    """
    path = Path(path)
    ds = xr.open_dataset(path, decode_times=True)
    try:
        df = normalize_ldl_dataset(ds, path)
    finally:
        ds.close()

    df["source_file"] = str(path)

    # Drop rows with NaN coordinates
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  {path.name}: dropped {dropped} rows with NaN coordinates")

    return df


def load_ldl_catalog(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load a list of LDL DWSD NetCDF files and return a concatenated DataFrame.

    Parameters
    ----------
    paths : list of paths to NetCDF files

    Returns
    -------
    Concatenated normalised DataFrame for all drifters.
    """
    frames = []
    errors = []

    for path in paths:
        try:
            df = load_ldl_file(path)
            frames.append(df)
        except Exception as e:
            errors.append({"file": Path(path), "error": str(e)})

    if errors:
        print(f"Warning: {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"  - {err['file'].name}: {err['error']}")

    if not frames:
        raise RuntimeError("No LDL files could be loaded.")

    result = pd.concat(frames, ignore_index=True)
    print(
        f"Loaded {len(frames)} LDL drifter(s) — "
        f"{len(result):,} total rows, "
        f"{result['tc_name'].nunique()} cyclone(s): "
        f"{', '.join(sorted(result['tc_name'].unique()))}"
    )
    return result


# ---------------------------------------------------------------------------
# Catalog builder (scan a directory)
# ---------------------------------------------------------------------------

def get_ldl_files(directory: str | Path) -> list[Path]:
    """
    Recursively find all NetCDF files in a directory.

    Parameters
    ----------
    directory : root directory to scan

    Returns
    -------
    Sorted list of Path objects.
    """
    return sorted(Path(directory).rglob("*.nc"))