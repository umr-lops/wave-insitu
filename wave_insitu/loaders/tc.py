"""
tc_loader.py
============
Loader for Tropical Cyclone (TC) track data from the CyclObs IFREMER API.

API endpoint : https://cyclobs.ifremer.fr/app/api/track
Source       : IBTrACS (by default)

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

Usage
-----
    from tc_loader import load_tc_tracks

    # From a reference DataFrame (uses its time bounds)
    tc_df = load_tc_tracks(reference_df=all_data)

    # With explicit date range
    tc_df = load_tc_tracks(min_date="2019-01-01", max_date="2024-12-31")
"""

import colorsys

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

API_URL    = "https://cyclobs.ifremer.fr/app/api/track"
API_SOURCE = "ibtracs"
API_FREQ   = 60 * 5      # 5-minute interpolation (seconds)


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def _fetch_tc_tracks(
    min_date: str,
    max_date: str,
    source: str = API_SOURCE,
    freq: int   = API_FREQ,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch TC track data from the CyclObs API and return a raw DataFrame.

    Parameters
    ----------
    min_date : start date string 'YYYY-MM-DD'
    max_date : end date string   'YYYY-MM-DD'
    source   : data source (default: 'ibtracs')
    freq     : interpolation frequency in seconds (default: 300)
    timeout  : request timeout in seconds

    Returns
    -------
    Raw DataFrame from the API CSV response.
    """
    params = {
        "source":   source,
        "freq":     freq,
        "min_date": min_date,
        "max_date": max_date,
    }
    url = f"{API_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    print(f"Fetching TC tracks: {url}")

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Saffir-Simpson category
# ---------------------------------------------------------------------------

# Wind speed thresholds in m/s converted from km/h
# km/h: <118 TD/TS, 118-153 cat1, 154-177 cat2, 178-209 cat3, 210-250 cat4, >250 cat5
# m/s boundaries (1 km/h = 1/3.6 m/s):
_SS_BOUNDS_MS = [
    (0.0,         118 / 3.6, 0),   # Tropical depression / storm
    (118 / 3.6,   154 / 3.6, 1),
    (154 / 3.6,   178 / 3.6, 2),
    (178 / 3.6,   210 / 3.6, 3),
    (210 / 3.6,   251 / 3.6, 4),
    (251 / 3.6,   9999.0,    5),
]


def wind_speed_to_category(ws_ms: float) -> int:
    """
    Return the Saffir-Simpson category (0-5) for a wind speed in m/s.
    Category 0 means tropical depression or tropical storm (below cat 1).
    Returns -1 for NaN/missing values.
    """
    if np.isnan(ws_ms):
        return -1
    for lo, hi, cat in _SS_BOUNDS_MS:
        if lo <= ws_ms < hi:
            return cat
    return 5  # above all thresholds


def assign_tc_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns to a TC track DataFrame:

    - 'point_category'  : Saffir-Simpson category at each individual point
    - 'max_category'    : maximum category reached over the full storm (per sid),
                          broadcast to all rows of that storm

    Parameters
    ----------
    df : TC track DataFrame with columns 'sid' and 'wind_speed' (m/s)

    Returns
    -------
    DataFrame with two additional integer columns.
    """
    df = df.copy()
    ws = df["wind_speed"].values.astype(float) if "wind_speed" in df.columns else np.full(len(df), np.nan)
    df["point_category"] = [wind_speed_to_category(w) for w in ws]

    # Max category per storm
    max_cat = df.groupby("sid")["point_category"].max().rename("max_category")
    df = df.join(max_cat, on="sid")

    return df


def _normalize_tc_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw API DataFrame into a clean, consistent format.

    Input columns (from API):
        date, lat, lon, sid, name, source, wind_speed (m/s), ...

    Output columns:
        time, latitude, longitude, sid, name, source, wind_speed, name_sid,
        point_category, max_category

    Parameters
    ----------
    raw : raw DataFrame from _fetch_tc_tracks()

    Returns
    -------
    Normalised DataFrame.
    """
    df = raw.copy()

    # Rename columns to canonical names
    rename_map = {
        "date":           "time",
        "lat":            "latitude",
        "lon":            "longitude",
        "wind_speed (m/s)": "wind_speed",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse time
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Ensure numeric coordinates and wind speed
    for col in ("latitude", "longitude", "wind_speed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a unique display name: "MILTON (2024283N18294)"
    if "sid" in df.columns and "name" in df.columns:
        df["name_sid"] = df["name"].str.title() + " (" + df["sid"].astype(str) + ")"
    else:
        df["name_sid"] = df.get("name", "Unknown")

    # Drop rows with missing coordinates or time
    df = df.dropna(subset=["time", "latitude", "longitude"])
    df = df.sort_values(["sid", "time"]).reset_index(drop=True)

    # Add Saffir-Simpson categories
    df = assign_tc_category(df)

    return df


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_tc_tracks(
    reference_df: pd.DataFrame | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    source: str = API_SOURCE,
    freq: int   = API_FREQ,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Load TC track data from the CyclObs API.

    Date bounds are inferred from reference_df if not provided explicitly.
    Explicit min_date / max_date always take precedence.

    Parameters
    ----------
    reference_df : DataFrame with a 'time' column used to infer date bounds
    min_date     : start date 'YYYY-MM-DD' (overrides reference_df)
    max_date     : end date   'YYYY-MM-DD' (overrides reference_df)
    source       : IBTrACS source string
    freq         : interpolation frequency in seconds
    timeout      : HTTP request timeout in seconds

    Returns
    -------
    Normalised TC track DataFrame with columns:
        time, latitude, longitude, sid, name, source, wind_speed, name_sid
    """
    # --- Resolve date bounds ---
    if min_date is None or max_date is None:
        if reference_df is None:
            raise ValueError(
                "Provide either reference_df or explicit min_date/max_date."
            )
        ref_times = pd.to_datetime(reference_df["time"])
        if min_date is None:
            min_date = ref_times.min().strftime("%Y-%m-%d")
        if max_date is None:
            max_date = ref_times.max().strftime("%Y-%m-%d")

    print(f"TC date range: {min_date} → {max_date}")

    # --- Fetch and normalise ---
    raw = _fetch_tc_tracks(min_date, max_date, source, freq, timeout)
    df  = _normalize_tc_df(raw)

    n_tc = df["sid"].nunique()
    print(
        f"Loaded {n_tc} TC track(s) — {len(df):,} points\n"
        #f"Storms: {', '.join(sorted(df['name_sid'].unique()))}"
    )
    return df