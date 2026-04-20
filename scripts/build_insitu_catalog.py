#!/usr/bin/env python3
"""
build_insitu_catalog.py
=======================
Build a complete in-situ observation catalog from multiple data sources.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

This script loads data from Saildrone, LDL DWSD, and KU-Buoys platforms,
merges them into a single catalog DataFrame, and saves to CSV.

Usage
-----
    # Default: use config/data_dirs.yaml
    python build_insitu_catalog.py --verbose
    
    # Custom config path
    python build_insitu_catalog.py --data-dirs /path/to/data_dirs.yaml --verbose
    
    # Override with individual paths
    python build_insitu_catalog.py \\
        --saildrone cmems:/path/to/cmems noaa:/path/to/noaa pimep:/path/to/pimep \\
        --ldl-dir /path/to/ldl \\
        --kub-dir /path/to/kub \\
        --verbose
"""

import argparse
from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import yaml

# Import loaders
from wave_insitu.loaders import saildrone, ldl, kub
from wave_insitu.utils import load_mapping


def load_data_dirs_config(config_path: str | Path) -> dict:
    """
    Load data directories from centralized YAML configuration.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML file with data source directories.
        Expected format:
        {
            "data_sources": {
                "saildrone": {
                    "cmems": "/path/to/cmems",
                    "noaa": "/path/to/noaa",
                    "pimep": "/path/to/pimep"
                },
                "ldl": {
                    "path": "/path/to/ldl"
                },
                "kub": {
                    "path": "/path/to/kub"
                }
            }
        }
    
    Returns
    -------
    dict
        Dictionary with keys: saildrone_dirs, ldl_dir, kub_dir
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if "data_sources" not in config:
        raise ValueError(f"Missing 'data_sources' key in {config_path}")
    
    sources = config["data_sources"]
    
    return {
        "saildrone_dirs": sources.get("saildrone", {}),
        "ldl_dir": sources.get("ldl", {}).get("path"),
        "kub_dir": sources.get("kub", {}).get("path"),
    }


def build_insitu_catalog(
    saildrone_dirs: Optional[dict] = None,
    ldl_dir: Optional[str | Path] = None,
    kub_dir: Optional[str | Path] = None,
    mapping_path: str | Path = "config/mapping.yaml",
    query_condition: str = "wave",
    output_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a complete in-situ observation catalog from multiple platforms.
    
    Parameters
    ----------
    saildrone_dirs : dict or None
        Dictionary of {provider: path} for saildrone data, or None to skip
    ldl_dir : str, Path, or None
        Path to LDL DWSD data directory, or None to skip
    kub_dir : str, Path, or None
        Path to KU-Buoys data directory, or None to skip
    mapping_path : str or Path
        Path to YAML mapping file (required for saildrone and KUB)
    query_condition : str
        Filter for saildrone data: "wave" (default), "wind", "wave and wind", "wave or wind"
    output_path : str, Path, or None
        Path to save catalog CSV. If None, catalog is not saved.
    verbose : bool
        Print progress information
    
    Returns
    -------
    pd.DataFrame
        Merged catalog with columns:
        time, latitude, longitude, platform_type, platform_id, provider, 
        name, source_file, tc_name, and variable-specific columns
    """
    frames = []
    
    if verbose:
        print("=" * 70)
        print("Building in-situ observation catalog")
        print("=" * 70)
    
    # --- Load mapping (needed for saildrone and KUB) ---
    mapping = load_mapping(mapping_path)
    if verbose:
        print(f"\nLoaded mapping from: {mapping_path}")
    
    # --- Load Saildrone data ---
    if saildrone_dirs:
        try:
            if verbose:
                print(f"\n--- Loading Saildrone data ---")
                print(f"Providers: {list(saildrone_dirs.keys())}")
            
            sd_df = saildrone.load_saildrone_from_dirs(
                saildrone_dirs,
                mapping,
                query_condition=query_condition,
                verbose=verbose
            )
            
            frames.append(sd_df)
            if verbose:
                print(f"✓ Loaded {len(sd_df)} Saildrone observations")
        
        except Exception as e:
            print(f"⚠ Error loading Saildrone data: {e}", file=sys.stderr)
    
    # --- Load LDL DWSD data ---
    if ldl_dir is not None:
        ldl_dir = Path(ldl_dir)
        if ldl_dir.exists():
            try:
                if verbose:
                    print(f"\n--- Loading LDL DWSD data ---")
                    print(f"Directory: {ldl_dir}")
                
                ldl_ncfiles = ldl.get_ldl_files(ldl_dir)
                
                if ldl_ncfiles:
                    ldl_df = ldl.load_ldl_catalog(ldl_ncfiles)
                    frames.append(ldl_df)
                    if verbose:
                        print(f"✓ Loaded {len(ldl_df)} LDL DWSD observations from {len(ldl_ncfiles)} files")
                else:
                    if verbose:
                        print("No LDL files found")
            
            except Exception as e:
                print(f"⚠ Error loading LDL data: {e}", file=sys.stderr)
        else:
            print(f"⚠ LDL directory not found: {ldl_dir}", file=sys.stderr)
    
    # --- Load KU-Buoys data ---
    if kub_dir is not None:
        kub_dir = Path(kub_dir)
        if kub_dir.exists():
            try:
                if verbose:
                    print(f"\n--- Loading KU-Buoys data ---")
                    print(f"Directory: {kub_dir}")
                
                kub_files = sorted(list(kub_dir.rglob("*.nc")))
                
                if kub_files:
                    kub_df = kub.load_kub_catalog(kub_files, mapping)
                    frames.append(kub_df)
                    if verbose:
                        print(f"✓ Loaded {len(kub_df)} KU-Buoys observations from {len(kub_files)} files")
                else:
                    if verbose:
                        print("No KU-Buoys files found")
            
            except Exception as e:
                print(f"⚠ Error loading KU-Buoys data: {e}", file=sys.stderr)
        else:
            print(f"⚠ KU-Buoys directory not found: {kub_dir}", file=sys.stderr)
    
    # --- Merge all platforms ---
    if not frames:
        raise RuntimeError("No data loaded from any platform!")
    
    if verbose:
        print(f"\n--- Merging data ---")
    
    merged_df = pd.concat(frames, ignore_index=True)
    
    # --- Reorder columns: metadata first, then variables ---
    common_cols = [
        'time', 'latitude', 'longitude', 'platform_type', 
        'platform_id', 'provider', 'name', 'source_file', 'tc_name'
    ]
    other_cols = [col for col in merged_df.columns if col not in common_cols]
    merged_df = merged_df[common_cols + other_cols]
    
    if verbose:
        print(f"Total: {len(merged_df)} observations from {merged_df['platform_id'].nunique()} platforms")
        print(f"Platform types: {merged_df['platform_type'].unique().tolist()}")
        print(f"Date range: {merged_df['time'].min()} to {merged_df['time'].max()}")
    
    # --- Save to CSV ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✓ Catalog saved to: {output_path}")
    
    return merged_df


def main():
    """Parse command-line arguments and build catalog."""
    parser = argparse.ArgumentParser(
        description="Build in-situ observation catalog from multiple sources"
    )
    
    parser.add_argument(
        "--data-dirs",
        type=str,
        default="config/data_dirs.yaml",
        help="Path to YAML file with all data source directories (default: config/data_dirs.yaml)"
    )
    
    parser.add_argument(
        "--saildrone",
        type=str,
        nargs="*",
        default=None,
        help="Override Saildrone directories (format: provider:/path provider:/path ...)"
    )
    
    parser.add_argument(
        "--ldl-dir",
        type=str,
        default=None,
        help="Override LDL DWSD directory"
    )
    
    parser.add_argument(
        "--kub-dir",
        type=str,
        default=None,
        help="Override KU-Buoys directory"
    )
    
    parser.add_argument(
        "--mapping",
        type=str,
        default="config/mapping.yaml",
        help="Path to YAML mapping file (default: config/mapping.yaml)"
    )
    
    parser.add_argument(
        "--query-condition",
        type=str,
        default="wave",
        choices=["wave", "wind", "wave and wind", "wave or wind"],
        help="Filter condition for saildrone data (default: wave)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="wave_obs_catalog.csv",
        help="Path to save output catalog CSV (default: wave_obs_catalog.csv)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, only return DataFrame"
    )
    
    args = parser.parse_args()
    
    # Determine data paths
    saildrone_dirs = None
    ldl_dir = args.ldl_dir
    kub_dir = args.kub_dir
    
    # If no individual paths are provided, use config file
    if not args.saildrone and ldl_dir is None and kub_dir is None:
        if Path(args.data_dirs).exists():
            config = load_data_dirs_config(args.data_dirs)
            saildrone_dirs = config["saildrone_dirs"]
            ldl_dir = config["ldl_dir"]
            kub_dir = config["kub_dir"]
        else:
            print(f"✗ Config file not found: {args.data_dirs}", file=sys.stderr)
            sys.exit(1)
    else:
        # Override with individual paths
        if args.saildrone:
            saildrone_dirs = {}
            for item in args.saildrone:
                if ":" in item:
                    provider, path = item.split(":", 1)
                    saildrone_dirs[provider] = path
                else:
                    print(f"✗ Invalid Saildrone format: {item}. Use 'provider:/path'", file=sys.stderr)
                    sys.exit(1)
    
    try:
        # Build catalog
        catalog_df = build_insitu_catalog(
            saildrone_dirs=saildrone_dirs,
            ldl_dir=ldl_dir,
            kub_dir=kub_dir,
            mapping_path=args.mapping,
            query_condition=args.query_condition,
            output_path=None if args.no_save else args.output,
            verbose=args.verbose,
        )
        
        if args.verbose:
            print("\n" + "=" * 70)
            print("Catalog building completed successfully")
            print("=" * 70)
    
    except Exception as e:
        print(f"✗ Error building catalog: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


def load_saildrone_dirs_config(config_path: str | Path) -> dict:
    """
    Load saildrone directories from YAML configuration.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML file with saildrone provider directories.
        Example format:
        {
            "cmems": "/path/to/cmems",
            "noaa": "/path/to/noaa",
            "pimep": "/path/to/pimep"
        }
    
    Returns
    -------
    dict
        Dictionary mapping provider names to directory paths
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict in {config_path}, got {type(config)}")
    
    return config


def build_insitu_catalog(
    saildrone_config: Optional[str | Path] = None,
    ldl_dir: Optional[str | Path] = None,
    kub_dir: Optional[str | Path] = None,
    mapping_path: str | Path = "config/mapping.yaml",
    query_condition: str = "wave",
    output_path: Optional[str | Path] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a complete in-situ observation catalog from multiple platforms.
    
    Parameters
    ----------
    saildrone_config : str, Path, or None
        Path to YAML file with saildrone provider directories, or None to skip
    ldl_dir : str, Path, or None
        Path to LDL DWSD data directory, or None to skip
    kub_dir : str, Path, or None
        Path to KU-Buoys data directory, or None to skip
    mapping_path : str or Path
        Path to YAML mapping file (required for saildrone and KUB)
    query_condition : str
        Filter for saildrone data: "wave" (default), "wind", "wave and wind", "wave or wind"
    output_path : str, Path, or None
        Path to save catalog CSV. If None, catalog is not saved.
    verbose : bool
        Print progress information
    
    Returns
    -------
    pd.DataFrame
        Merged catalog with columns:
        time, latitude, longitude, platform_type, platform_id, provider, 
        name, source_file, tc_name, and variable-specific columns
    """
    frames = []
    
    if verbose:
        print("=" * 70)
        print("Building in-situ observation catalog")
        print("=" * 70)
    
    # --- Load mapping (needed for saildrone and KUB) ---
    mapping = load_mapping(mapping_path)
    if verbose:
        print(f"\nLoaded mapping from: {mapping_path}")
    
    # --- Load Saildrone data ---
    if saildrone_config is not None:
        saildrone_config = Path(saildrone_config)
        if saildrone_config.exists():
            try:
                if verbose:
                    print(f"\n--- Loading Saildrone data ---")
                    print(f"Config: {saildrone_config}")
                
                sddirs = load_saildrone_dirs_config(saildrone_config)
                
                sd_df = saildrone.load_saildrone_from_dirs(
                    sddirs,
                    mapping,
                    query_condition=query_condition,
                    verbose=verbose
                )
                
                frames.append(sd_df)
                if verbose:
                    print(f"✓ Loaded {len(sd_df)} Saildrone observations")
            
            except Exception as e:
                print(f"⚠ Error loading Saildrone data: {e}", file=sys.stderr)
        else:
            print(f"⚠ Saildrone config not found: {saildrone_config}", file=sys.stderr)
    
    # --- Load LDL DWSD data ---
    if ldl_dir is not None:
        ldl_dir = Path(ldl_dir)
        if ldl_dir.exists():
            try:
                if verbose:
                    print(f"\n--- Loading LDL DWSD data ---")
                    print(f"Directory: {ldl_dir}")
                
                ldl_ncfiles = ldl.get_ldl_files(ldl_dir)
                
                if ldl_ncfiles:
                    ldl_df = ldl.load_ldl_catalog(ldl_ncfiles)
                    frames.append(ldl_df)
                    if verbose:
                        print(f"✓ Loaded {len(ldl_df)} LDL DWSD observations from {len(ldl_ncfiles)} files")
                else:
                    if verbose:
                        print("No LDL files found")
            
            except Exception as e:
                print(f"⚠ Error loading LDL data: {e}", file=sys.stderr)
        else:
            print(f"⚠ LDL directory not found: {ldl_dir}", file=sys.stderr)
    
    # --- Load KU-Buoys data ---
    if kub_dir is not None:
        kub_dir = Path(kub_dir)
        if kub_dir.exists():
            try:
                if verbose:
                    print(f"\n--- Loading KU-Buoys data ---")
                    print(f"Directory: {kub_dir}")
                
                kub_files = sorted(list(kub_dir.rglob("*.nc")))
                
                if kub_files:
                    kub_df = kub.load_kub_catalog(kub_files, mapping)
                    frames.append(kub_df)
                    if verbose:
                        print(f"✓ Loaded {len(kub_df)} KU-Buoys observations from {len(kub_files)} files")
                else:
                    if verbose:
                        print("No KU-Buoys files found")
            
            except Exception as e:
                print(f"⚠ Error loading KU-Buoys data: {e}", file=sys.stderr)
        else:
            print(f"⚠ KU-Buoys directory not found: {kub_dir}", file=sys.stderr)
    
    # --- Merge all platforms ---
    if not frames:
        raise RuntimeError("No data loaded from any platform!")
    
    if verbose:
        print(f"\n--- Merging data ---")
    
    merged_df = pd.concat(frames, ignore_index=True)
    
    # --- Reorder columns: metadata first, then variables ---
    common_cols = [
        'time', 'latitude', 'longitude', 'platform_type', 
        'platform_id', 'provider', 'name', 'source_file', 'tc_name'
    ]
    other_cols = [col for col in merged_df.columns if col not in common_cols]
    merged_df = merged_df[common_cols + other_cols]
    
    if verbose:
        print(f"Total: {len(merged_df)} observations from {merged_df['platform_id'].nunique()} platforms")
        print(f"Platform types: {merged_df['platform_type'].unique().tolist()}")
        print(f"Date range: {merged_df['time'].min()} to {merged_df['time'].max()}")
    
    # --- Save to CSV ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✓ Catalog saved to: {output_path}")
    
    return merged_df


def main():
    """Parse command-line arguments and build catalog."""
    parser = argparse.ArgumentParser(
        description="Build in-situ observation catalog from multiple sources"
    )
    
    parser.add_argument(
        "--saildrone-config",
        type=str,
        default="config/saildrone_dirs.yaml",
        help="Path to YAML file with saildrone provider directories (default: config/saildrone_dirs.yaml)"
    )
    
    parser.add_argument(
        "--ldl-dir",
        type=str,
        default="/scale/user/egauvrit/data/insitu/DWSD",
        help="Path to LDL DWSD data directory"
    )
    
    parser.add_argument(
        "--kub-dir",
        type=str,
        default="/scale/user/egauvrit/data/insitu/KUB/SWH",
        help="Path to KU-Buoys data directory"
    )
    
    parser.add_argument(
        "--mapping",
        type=str,
        default="config/mapping.yaml",
        help="Path to YAML mapping file (default: config/mapping.yaml)"
    )
    
    parser.add_argument(
        "--query-condition",
        type=str,
        default="wave",
        choices=["wave", "wind", "wave and wind", "wave or wind"],
        help="Filter condition for saildrone data (default: wave)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="wave_obs_catalog.csv",
        help="Path to save output catalog CSV (default: wave_obs_catalog.csv)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to file, only return DataFrame"
    )
    
    args = parser.parse_args()
    
    try:
        # Build catalog
        catalog_df = build_insitu_catalog(
            saildrone_config=args.saildrone_config if Path(args.saildrone_config).exists() else None,
            ldl_dir=args.ldl_dir,
            kub_dir=args.kub_dir,
            mapping_path=args.mapping,
            query_condition=args.query_condition,
            output_path=None if args.no_save else args.output,
            verbose=args.verbose,
        )
        
        if args.verbose:
            print("\n" + "=" * 70)
            print("Catalog building completed successfully")
            print("=" * 70)
    
    except Exception as e:
        print(f"✗ Error building catalog: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
