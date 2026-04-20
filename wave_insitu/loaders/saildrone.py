"""
saildrone.py
============
Loader for Saildrone autonomous platform NetCDF files.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)
"""

import re
import xarray as xr
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from netCDF4 import Dataset

from ..utils import load_mapping, build_reverse_lookup


# ---------------------------------------------------------------------------
# File discovery and catalog building
# ---------------------------------------------------------------------------

def list_variables(files: list) -> list:
    """
    List all unique variable names from a collection of NetCDF files.
    
    Parameters
    ----------
    files : list of paths
    
    Returns
    -------
    Sorted list of variable names found
    """
    vars_found = set()
    for f in tqdm(files):
        try:
            ds = xr.open_dataset(f)
            vars_found.update(ds.data_vars)
            ds.close()
        except Exception:
            pass
    return sorted(vars_found)


def get_files(provider: str, sddirs: dict) -> list:
    """
    Get all .nc files from a provider's directory.
    
    Parameters
    ----------
    provider : provider name (key in sddirs dict)
    sddirs : dictionary mapping provider -> directory path
    
    Returns
    -------
    Sorted list of file paths
    """
    if provider not in sddirs:
        raise ValueError(f"Provider '{provider}' not found in sddirs")
    return sorted(list(Path(sddirs[provider]).rglob("*.nc")))


def build_file_catalog(sddirs: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Build a catalog DataFrame of all .nc files from multiple providers.
    
    Parameters
    ----------
    sddirs : dictionary mapping provider -> directory path
    verbose : print progress info
    
    Returns
    -------
    DataFrame with columns: provider, path, name
    """
    catalog = []
    for provider in sddirs:
        if verbose:
            print(f"Scanning {provider}...")
        try:
            files = get_files(provider, sddirs)
            for f in files:
                catalog.append({
                    "provider": provider,
                    "path": f,
                    "name": f.name
                })
        except Exception as e:
            if verbose:
                print(f"  Error scanning {provider}: {e}")
            continue
    
    df = pd.DataFrame(catalog)
    if verbose and not df.empty:
        print(f"Found {len(df)} files from {len(sddirs)} provider(s)")
    return df


def build_alias_sets(mapping: dict) -> tuple[set, set]:
    """
    Build alias sets for wave and wind variables from mapping.
    
    Parameters
    ----------
    mapping : loaded YAML mapping dict
    
    Returns
    -------
    (wave_alias, wind_alias) : two sets of lowercase variable aliases
    """
    wave_alias = set()
    wave_alias |= {
        a.lower()
        for a in mapping.get("variables", {}).get("significant_wave_height", [])
    }
    wave_alias |= {
        a.lower()
        for a in mapping.get("variables", {}).get("dominant_wave_period", [])
    }

    wind_alias = set()
    wind_alias |= {
        a.lower() for a in mapping.get("variables", {}).get("wind_speed", [])
    }
    wind_alias |= {
        a.lower() for a in mapping.get("variables", {}).get("wind_direction", [])
    }

    return wave_alias, wind_alias


def check_wind_wave(path: str | Path, mapping: dict) -> tuple[bool, bool]:
    """
    Check if a NetCDF file contains wave and/or wind variables.
    
    Parameters
    ----------
    path : path to NetCDF file
    mapping : loaded YAML mapping dict
    
    Returns
    -------
    (has_wave, has_wind) : tuple of booleans
    """
    wave_alias = set(mapping.get("variables", {}).get("significant_wave_height", []))
    wind_alias = set(mapping.get("variables", {}).get("wind_speed", []))
    
    try:
        with Dataset(path, "r") as ds:
            vars_lower = {v.lower() for v in ds.variables.keys()}

        has_wave = bool(vars_lower & wave_alias)
        has_wind = bool(vars_lower & wind_alias)

        return has_wave, has_wind

    except Exception:
        return False, False





# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------

def _flatten_to_1d(arr: np.ndarray, n: int) -> np.ndarray:
    """Ensure a 1D array of length n regardless of the input shape."""
    if arr.ndim == 0:
        return np.full(n, arr.item())
    arr = arr.squeeze()
    if arr.ndim == 0:
        return np.full(n, arr.item())
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr[:n]


# ---------------------------------------------------------------------------
# Dimension normalisation
# ---------------------------------------------------------------------------

def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalise dataset dimensions to a strictly 1D format along the time axis.
 
    Handled formats:
      A) (TIME,) + optional DEPTH dimension          -> CMEMS
      B) (row,)                                      -> NOAA / PIMEP gen_5 old
      C) (trajectory, obs), trajectory of size 1     -> NOAA gen_5/6 CF-trajectory
 
    For DEPTH-like dimensions: apply nanmean across depth instead of isel(0),
    so that variables with NaN at the surface level are not lost.
 
    Returns a Dataset where all variables have shape (n,).
    """
    dim_sizes = dict(ds.sizes)
 
    # --- Collapse DEPTH-like dimensions with nanmean ---
    # This preserves wave height values that may be NaN at depth index 0
    DEPTH_DIMS = {"depth", "deph", "z"}
    for d in list(dim_sizes):
        if d.lower() in DEPTH_DIMS:
            ds = ds.mean(dim=d, skipna=True, keep_attrs=True)
            dim_sizes = dict(ds.sizes)
            break  # only one depth dim expected
 
    # Collapse remaining structural dimensions (string_length, etc.) with isel
    STRUCTURAL_DIMS = {"string_length", "strlen", "nchar"}
    to_reduce = {d: 0 for d in dim_sizes if d.lower() in STRUCTURAL_DIMS}
    if to_reduce:
        ds = ds.isel(to_reduce)
        dim_sizes = dict(ds.sizes)
 
    # --- Format C: CF-trajectory (trajectory, obs) ---
    traj_dim = next((d for d in dim_sizes if d.lower() == "trajectory"), None)
    obs_dim  = next((d for d in dim_sizes if d.lower() == "obs"), None)
 
    if traj_dim and obs_dim:
        n_traj = dim_sizes[traj_dim]
        if n_traj == 1:
            ds = ds.isel({traj_dim: 0})
        else:
            # Rare: multiple drones in a single file -> flatten
            ds = ds.stack({"_flat": (traj_dim, obs_dim)}).reset_index("_flat")
 
    return ds


# ---------------------------------------------------------------------------
# Saildrone ID extraction
# ---------------------------------------------------------------------------

def _extract_sd_id(ds: xr.Dataset, mapping: dict) -> str:
    """
    Extract the numeric saildrone ID from dataset attributes.

    Sources consulted in order:
      1. Global attributes whose key matches mapping["attributes"]["sd_id"]
      2. Variable/coordinate named "trajectory" (fallback for formats B/C)

    Normalisation examples:
      "Saildrone sd-1058"  -> "1058"
      "sd1021"             -> "1021"
      1021.0               -> "1021"
      "1021"               -> "1021"
    """
    id_aliases = [
        a.lower()
        for a in mapping.get("attributes", {}).get("sd_id", [])
    ]

    raw_id = None

    # 1. Search global attributes
    for key, val in ds.attrs.items():
        if key.lower() in id_aliases:
            raw_id = str(val).strip()
            break

    # 2. Fallback: "trajectory" variable/coord (formats B and C before dim normalisation)
    if raw_id is None:
        all_names_lower = {
            nm.lower(): nm for nm in list(ds.coords) + list(ds.data_vars)
        }
        if "trajectory" in all_names_lower:
            raw = ds[all_names_lower["trajectory"]].values.squeeze()
            raw_id = str(raw.item()) if raw.ndim == 0 else str(raw[0])

    if raw_id is None:
        return "unknown"

    # Extract first group of >= 4 consecutive digits
    # Covers: "Saildrone sd-1058", "sd1021", "1021.0", "platform_1036"
    match = re.search(r"\d{4,}", raw_id)
    if match:
        return match.group(0)

    # Last resort: float -> int conversion
    try:
        return str(int(float(raw_id)))
    except (ValueError, TypeError):
        return raw_id.strip()


# ---------------------------------------------------------------------------
# Trajectory segmentation
# ---------------------------------------------------------------------------

def _haversine_consecutive(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute haversine distances (km) between consecutive (lat, lon) pairs.
    Returns an array of length (n-1).
    """
    R = 6371.0
    lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
    lon1, lon2 = np.radians(lons[:-1]), np.radians(lons[1:])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def split_trajectory_segments(
    df: pd.DataFrame,
    base_name: str,
    max_distance_km: float = 100.0,
    min_segment_points: int = 100,
) -> pd.DataFrame:
    """
    Split a single-file trajectory DataFrame into segments based on spatial jumps.

    A new segment starts whenever the distance between two consecutive points
    exceeds max_distance_km. Segments with fewer than min_segment_points rows
    are discarded.

    The 'name' column is updated to '<base_name>-seg1', '<base_name>-seg2', etc.
    If only one valid segment exists, 'name' stays as '<base_name>' (no suffix).

    Parameters
    ----------
    df : DataFrame sorted by time, with latitude and longitude columns
    base_name : original file stem used as the segment name prefix
    max_distance_km : distance threshold above which a jump is detected
    min_segment_points : minimum number of points to keep a segment

    Returns
    -------
    DataFrame with updated 'name' column and isolated/short segments removed.
    """
    df = df.copy().reset_index(drop=True)

    lats = df["latitude"].values
    lons = df["longitude"].values

    # Compute consecutive distances; prepend 0 so index aligns with df rows
    dists = _haversine_consecutive(lats, lons)
    jump  = np.concatenate([[False], dists > max_distance_km])

    # Assign a segment index that increments at each jump
    segment_ids = np.cumsum(jump)
    df["_segment"] = segment_ids

    # Filter out segments that are too short
    seg_counts = df["_segment"].value_counts()
    valid_segs  = seg_counts[seg_counts >= min_segment_points].index
    df = df[df["_segment"].isin(valid_segs)].copy()

    if df.empty:
        return df

    # Re-number segments contiguously after filtering
    unique_segs = sorted(df["_segment"].unique())
    seg_remap   = {old: new for new, old in enumerate(unique_segs, start=1)}
    df["_segment"] = df["_segment"].map(seg_remap)

    n_segs = df["_segment"].nunique()

    if n_segs == 1:
        # No split needed: keep the original name as-is
        df["name"] = base_name
    else:
        df["name"] = base_name + "-seg" + df["_segment"].astype(str)

    df = df.drop(columns=["_segment"])

    if n_segs > 1:
        n_jumps = int(jump.sum())
        print(
            f"  {base_name}: {n_jumps} jump(s) detected -> "
            f"{n_segs} segments kept"
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset normalisation
# ---------------------------------------------------------------------------

def normalize_dataset(
    ds: xr.Dataset,
    reverse_lookup: dict,
    mapping: dict,
) -> pd.DataFrame:
    """
    Normalise a saildrone xr.Dataset into a uniform DataFrame.

    Guaranteed columns  : time, latitude, longitude, platform_id,
                          tc_name, provider, platform_type
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

    # --- 2. Extract saildrone ID before dimension normalisation ---
    # (global attributes and the trajectory variable are still intact here)
    sd_id = _extract_sd_id(ds, mapping)

    # --- 3. Normalise dimensions ---
    ds = _normalize_dims(ds)

    # --- 4. Extract base coordinates ---
    time = ds[time_col].values
    n    = len(time)
    lat  = _flatten_to_1d(ds[lat_col].values, n)
    lon  = _flatten_to_1d(ds[lon_col].values, n)

    # --- 5. Build base DataFrame ---
    df = pd.DataFrame({
        "time":         pd.to_datetime(time),
        "latitude":     lat.astype(float),
        "longitude":    lon.astype(float),
        "platform_id":  sd_id,
        "tc_name":      "unknown",
        "platform_type": "saildrone",
    })

    # --- 6. Add variables of interest ---
    TARGET_VARIABLES = {
        "significant_wave_height",
        "dominant_wave_period",
        "wind_speed",
        "wind_direction",
    }

    for var_name in ds.data_vars:
        canonical = reverse_lookup.get(var_name.lower())
        if canonical in TARGET_VARIABLES and canonical not in df.columns:
            df[canonical] = _flatten_to_1d(ds[var_name].values, n).astype(float)

    # --- 7. Sort by time ---
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_saildrone_file(
    path: str | Path,
    mapping: dict,
    max_distance_km: float = 100.0,
    min_segment_points: int = 100,
) -> pd.DataFrame:
    """
    Load a saildrone NetCDF file and return a normalised, segmented DataFrame.

    Parameters
    ----------
    path : path to the NetCDF file
    mapping : loaded YAML mapping dict
    max_distance_km : spatial jump threshold for trajectory segmentation
    min_segment_points : minimum points required to retain a segment
    """
    path = Path(path)
    base_name = path.stem
    reverse_lookup = build_reverse_lookup(mapping)

    ds = xr.open_dataset(path, decode_times=True)
    try:
        df = normalize_dataset(ds, reverse_lookup, mapping)
    finally:
        ds.close()

    df["source_file"] = str(path)
    df["name"]        = base_name   # temporary; overwritten by segmentation
    df["provider"]    = "Saildrone"

    # Drop NaN coordinates before computing distances
    df = df.dropna(subset=["latitude", "longitude"])

    # Segment the trajectory
    df = split_trajectory_segments(
        df,
        base_name=base_name,
        max_distance_km=max_distance_km,
        min_segment_points=min_segment_points,
    )

    # Drop segments where significant_wave_height is entirely NaN
    if "significant_wave_height" in df.columns and not df.empty:
        hs_valid = df.groupby("name")["significant_wave_height"].apply(
            lambda s: s.notna().any()
        )
        valid_names = hs_valid[hs_valid].index
        n_dropped = df["name"].nunique() - len(valid_names)
        if n_dropped > 0:
            print(f"  {base_name}: {n_dropped} segment(s) dropped (Hs all-NaN)")
        df = df[df["name"].isin(valid_names)].copy()

    return df


def load_saildrone_catalog(
    catalog_df: pd.DataFrame,
    mapping: dict,
    max_distance_km: float = 100.0,
    min_segment_points: int = 100,
) -> pd.DataFrame:
    """
    Load all files from a catalog DataFrame and return a normalised,
    concatenated and segmented DataFrame.

    Parameters
    ----------
    catalog_df : DataFrame with at least 'path' and 'provider' columns
    mapping : loaded YAML mapping dict
    max_distance_km : spatial jump threshold for trajectory segmentation
    min_segment_points : minimum points required to retain a segment
    """
    frames = []
    errors = []

    for _, row in catalog_df.iterrows():
        try:
            df = load_saildrone_file(
                row["path"], mapping, max_distance_km, min_segment_points
            )
            df["provider"] = row["provider"]
            frames.append(df)
        except Exception as e:
            errors.append({"file": Path(row["path"]), "error": str(e)})

    if errors:
        print(f"Warning: {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"  - {err['file'].name}: {err['error']}")

    if not frames:
        raise RuntimeError("No files could be loaded.")

    return pd.concat(frames, ignore_index=True)


def load_saildrone_from_dirs(
    sddirs: dict,
    mapping: dict,
    query_condition: str | None = "wave",
    max_distance_km: float = 100.0,
    min_segment_points: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load saildrone data from multiple directories, with optional filtering.
    
    This function:
    1. Discovers all .nc files from multiple providers (via sddirs dict)
    2. Removes duplicate filenames
    3. Checks each file for wave/wind variables
    4. Applies optional query filter (e.g., "wave" to keep only files with wave data)
    5. Loads and normalises all selected files
    
    Parameters
    ----------
    sddirs : dict
        Dictionary mapping provider names to directory paths.
        Example: {
            "cmems": "/path/to/cmems",
            "noaa": "/path/to/noaa",
            "pimep": "/path/to/pimep"
        }
    mapping : dict
        Loaded YAML mapping dict (from load_mapping())
    query_condition : str or None, optional
        Filter condition to apply to the catalog. Options:
        - "wave" : keep only files with wave measurements (default)
        - "wind" : keep only files with wind measurements  
        - "wave and wind" : keep only files with both
        - "wave or wind" : keep only files with either
        - None : keep all files (no filtering)
    max_distance_km : float, optional
        Spatial jump threshold for trajectory segmentation (default: 100 km)
    min_segment_points : int, optional
        Minimum points required to retain a segment (default: 100)
    verbose : bool, optional
        Print progress information (default: False)
    
    Returns
    -------
    pd.DataFrame
        Normalised, concatenated saildrone data with columns:
        time, latitude, longitude, platform_id, tc_name, provider, 
        platform_type, name, source_file, and optional measured variables
    
    Examples
    --------
    >>> sddirs = {
    ...     "cmems": "/home/ref-copernicus-insitu/.../SD",
    ...     "noaa": "/home/datawork-cersat-public/provider/noaa/insitu/saildrone",
    ...     "pimep": "/home/datawork-cersat-public/project/pimep/data/saildrone",
    ... }
    >>> mapping = load_mapping("mapping.yaml")
    >>> sd_df = load_saildrone_from_dirs(
    ...     sddirs, mapping, query_condition="wave", verbose=True
    ... )
    """
    if verbose:
        print(f"Loading saildrone data from {len(sddirs)} provider(s)...")
    
    # --- Build catalog from directories ---
    df = build_file_catalog(sddirs, verbose=verbose)
    
    if df.empty:
        raise RuntimeError("No files found in any provider directory")
    
    # --- Remove duplicate filenames, keep first ---
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="name", keep='first')
    if verbose and len(df) < n_before_dedup:
        print(f"Removed {n_before_dedup - len(df)} duplicate file(s)")
    
    # --- Check for wave/wind variables ---
    if verbose:
        print("Checking files for wave/wind variables...")
    
    df[["wave", "wind"]] = df["path"].apply(
        lambda p: check_wind_wave(p, mapping)
    ).apply(pd.Series)
    
    # --- Apply query filter ---
    if query_condition is not None:
        n_before_query = len(df)
        if query_condition.lower() == "wave":
            df = df.query("wave")
        elif query_condition.lower() == "wind":
            df = df.query("wind")
        elif query_condition.lower() == "wave and wind":
            df = df.query("wave and wind")
        elif query_condition.lower() == "wave or wind":
            df = df.query("wave or wind")
        else:
            raise ValueError(
                f"Unknown query_condition: {query_condition}. "
                "Must be one of: 'wave', 'wind', 'wave and wind', 'wave or wind', or None"
            )
        
        if verbose and len(df) < n_before_query:
            print(f"Query '{query_condition}' removed {n_before_query - len(df)} file(s)")
    
    if df.empty:
        raise RuntimeError(f"No files match query condition: {query_condition}")
    
    if verbose:
        print(f"Loading {len(df)} files...")
    
    # --- Load all selected files ---
    result_df = load_saildrone_catalog(
        df, mapping, max_distance_km, min_segment_points
    )
    
    if verbose:
        print(f"✓ Loaded {len(result_df)} observations from {result_df['name'].nunique()} segments")
    
    return result_df
