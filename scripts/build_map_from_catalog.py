"""
Build an interactive map from a merged in-situ observation catalog CSV.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

This script loads a consolidated catalog of ocean wave observations
(Saildrone, LDL DWSD, KU-Buoys) and generates an interactive Folium map
with tropical cyclone track overlays.

Usage:
  python build_map_from_catalog.py
  python build_map_from_catalog.py --catalog /path/to/catalog.csv --output map.html
"""

import argparse
from pathlib import Path
from importlib import reload

import pandas as pd
import numpy as np
import yaml

# Import from the wave_insitu package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wave_insitu.loaders import saildrone, ldl, kub, tc
from wave_insitu.visualization import map as ism


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Determine project directory and defaults
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_CATALOG_PATH = "/scale/user/egauvrit/data/insitu/wave_obs_catalog.csv"
DEFAULT_OUTPUT_PATH = str(PROJECT_DIR / "insitu_map.html")
DEFAULT_WIND_COLORMAP = str(PROJECT_DIR / "config" / "wind_faozi.cpt")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_insitu_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Load the merged in-situ observation catalog.
    
    Parameters
    ----------
    catalog_path : str
        Path to the consolidated CSV catalog file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, latitude, longitude, platform_type,
        platform_id, provider, name, source_file, tc_name, + measured variables
    """
    print(f"Loading in-situ catalog from: {catalog_path}")
    
    # Use low_memory=False to avoid dtype warning for mixed-type columns
    df = pd.read_csv(catalog_path, low_memory=False)
    
    # Convert time column to datetime, handling mixed formats
    # Some timestamps have microseconds, others don't
    # Note: Keep timezone-naive to match TC track data format
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    
    # Ensure tc_name is string type (may have NaN or numeric values)
    df["tc_name"] = df["tc_name"].astype(str)
    
    print(f"  Loaded {len(df):,} observations")
    print(f"  Date range: {df['time'].min().date()} to {df['time'].max().date()}")
    return df


def separate_by_platform_type(insitu_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Separate observations by platform type.
    
    Parameters
    ----------
    insitu_df : pd.DataFrame
        Merged in-situ observations DataFrame
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: 'saildrone', 'dwsd', 'spotter'
        Each value is a filtered DataFrame for that platform type
    """
    platform_types = {
        "saildrone": "saildrone",
        "dwsd": "dwsd",
        "spotter": "spotter",
    }
    
    separated = {}
    for key, ptype in platform_types.items():
        df_subset = insitu_df.query(f"platform_type == '{ptype}'").copy()
        n_obs = len(df_subset)
        n_tracks = df_subset["name"].nunique()
        print(f"  {ptype.upper():8} : {n_obs:,} observations from {n_tracks} track(s)")
        separated[key] = df_subset
    
    return separated


def load_tc_tracks_for_period(insitu_df: pd.DataFrame, min_date: str = None, max_date: str = None) -> pd.DataFrame:
    """
    Load TC track data for the time period covered by in-situ observations.
    
    Parameters
    ----------
    insitu_df : pd.DataFrame
        Merged in-situ observations (used to determine date range if not specified)
    min_date : str, optional
        Minimum date in format 'YYYY-MM-DD'. If None, uses minimum from insitu_df.
    max_date : str, optional
        Maximum date in format 'YYYY-MM-DD'. If None, uses maximum from insitu_df.
        
    Returns
    -------
    pd.DataFrame
        TC track DataFrame with columns: time, latitude, longitude, sid, name, name_sid, wind_speed
    """
    if min_date is None:
        min_date = insitu_df["time"].min().strftime("%Y-%m-%d")
    if max_date is None:
        max_date = insitu_df["time"].max().strftime("%Y-%m-%d")
    
    print(f"Loading TC tracks for period: {min_date} to {max_date}")
    tc_df = tc.load_tc_tracks(min_date=min_date, max_date=max_date)
    print(f"  Loaded {len(tc_df):,} TC track points")
    return tc_df


# ---------------------------------------------------------------------------
# Map Building
# ---------------------------------------------------------------------------

def build_map_from_catalog(
    catalog_path: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    wind_colormap_path: str = DEFAULT_WIND_COLORMAP,
    ws_vmin: float = 0,
    ws_vmax: float = 136,
    max_points_per_track: int = 800,
) -> None:
    """
    Main pipeline: Load catalog, separate data, and build interactive map.
    
    Parameters
    ----------
    catalog_path : str
        Path to the merged in-situ catalog CSV
    output_path : str
        Output path for the generated HTML map (default: insitu_map_from_catalog.html)
    wind_colormap_path : str
        Path to GMT .cpt file for TC wind speed colormap
    ws_vmin : float
        Wind speed lower bound for colormap normalization (default: 0 m/s)
    ws_vmax : float
        Wind speed upper bound for colormap normalization (default: 136 m/s)
    max_points_per_track : int
        Maximum number of points to render per trajectory (for HTML size management)
    """
    print("\n" + "=" * 70)
    print("BUILDING IN-SITU OBSERVATION MAP FROM CATALOG")
    print("=" * 70)
    
    # ========== STEP 1: Load in-situ observations ==========
    print("\n[1/4] Loading in-situ observations...")
    insitu_df = load_insitu_catalog(catalog_path)
    
    # ========== STEP 2: Separate by platform type ==========
    print("\n[2/4] Separating observations by platform type...")
    separated = separate_by_platform_type(insitu_df)
    
    saildrone_df = separated["saildrone"]
    ldl_df = separated["dwsd"]
    kub_df = separated["spotter"]
    
    # ========== STEP 3: Load TC tracks ==========
    print("\n[3/4] Loading tropical cyclone track data...")
    tc_df = load_tc_tracks_for_period(insitu_df)
    
    # ========== STEP 4: Build map ==========
    print("\n[4/4] Building interactive Folium map...")
    print(f"  Wind colormap: {wind_colormap_path}")
    print(f"  Max points per track: {max_points_per_track}")
    print(f"  Output: {output_path}")
    
    m = ism.build_insitu_map(
        saildrone_df=saildrone_df,
        ldl_df=ldl_df,
        tc_df=tc_df,
        kub_df=kub_df,
        cpt_path=wind_colormap_path,
        output_path=output_path,
        ws_vmin=ws_vmin,
        ws_vmax=ws_vmax,
        max_points_per_track=max_points_per_track,
    )
    
    print("\n" + "=" * 70)
    print("✓ Map generation complete!")
    print(f"  Map saved to: {output_path}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------------

def main():
    """Parse command-line arguments and execute the pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate an interactive Folium map from a merged in-situ observation catalog CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default catalog and output paths
  python build_map_from_catalog.py
  
  # Specify custom catalog and output paths
  python build_map_from_catalog.py \\
    --catalog /path/to/catalog.csv \\
    --output /path/to/map.html
  
  # Customize wind speed colormap scale
  python build_map_from_catalog.py \\
    --ws-vmin 0 --ws-vmax 150
    """
    )
    
    parser.add_argument(
        "--catalog",
        type=str,
        default=DEFAULT_CATALOG_PATH,
        help=f"Path to merged in-situ observation catalog CSV (default: {DEFAULT_CATALOG_PATH})",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for the generated HTML map (default: {DEFAULT_OUTPUT_PATH})",
    )
    
    parser.add_argument(
        "--wind-colormap",
        type=str,
        default=DEFAULT_WIND_COLORMAP,
        help=f"Path to GMT .cpt colormap file for TC wind speed (default: {DEFAULT_WIND_COLORMAP})",
    )
    
    parser.add_argument(
        "--ws-vmin",
        type=float,
        default=0,
        help="Wind speed lower bound for colormap normalization (m/s, default: 0)",
    )
    
    parser.add_argument(
        "--ws-vmax",
        type=float,
        default=136,
        help="Wind speed upper bound for colormap normalization (m/s, default: 136)",
    )
    
    parser.add_argument(
        "--max-points",
        type=int,
        default=800,
        help="Maximum points per trajectory for HTML rendering (default: 2000)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Error: Catalog file not found: {args.catalog}")
        exit(1)
    
    # Build map
    build_map_from_catalog(
        catalog_path=str(catalog_path),
        output_path=args.output,
        wind_colormap_path=args.wind_colormap,
        ws_vmin=args.ws_vmin,
        ws_vmax=args.ws_vmax,
        max_points_per_track=args.max_points,
    )


if __name__ == "__main__":
    main()
