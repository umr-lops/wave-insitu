"""
utils.py
========
Common utility functions shared across wave_insitu loaders and visualization.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)
"""

import yaml
from typing import Dict, Tuple, Set


def load_mapping(path: str) -> dict:
    """
    Load YAML mapping file for variable/coordinate aliases.
    
    Parameters
    ----------
    path : str
        Path to YAML mapping file
    
    Returns
    -------
    dict
        Loaded mapping dictionary
    """
    with open(path) as f:
        return yaml.safe_load(f)


def build_reverse_lookup(mapping: dict) -> dict:
    """
    Build a dict mapping alias -> canonical name for variables and coordinates.
    
    This creates a reverse lookup from variable aliases to their canonical names,
    enabling flexible matching of different naming conventions in input files.
    
    Example:
        mapping = {
            "variables": {
                "significant_wave_height": ["Hs", "swh", "vavh"],
                "wind_speed": ["wind", "wind_speed", "ws"]
            }
        }
        
        reverse = build_reverse_lookup(mapping)
        # Result: {
        #     "hs": "significant_wave_height",
        #     "swh": "significant_wave_height",
        #     "vavh": "significant_wave_height",
        #     "wind": "wind_speed",
        #     "wind_speed": "wind_speed",
        #     "ws": "wind_speed"
        # }
    
    Parameters
    ----------
    mapping : dict
        Mapping dictionary with "variables" and/or "coordinates" sections
    
    Returns
    -------
    dict
        Reverse lookup: {alias_lower: canonical_name}
    """
    reverse = {}
    for section in ("variables", "coordinates"):
        for canonical, aliases in mapping.get(section, {}).items():
            for alias in aliases:
                reverse[alias.lower()] = canonical
    return reverse
