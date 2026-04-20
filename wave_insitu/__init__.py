"""
wave-insitu: Tools for processing and visualizing in-situ ocean wave observations.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

This package provides loaders for multiple ocean observation platforms (Saildrone,
LDL DWSD, KU-Buoys) and tools for generating interactive maps.

Modules:
  - wave_insitu.loaders : Data loaders for different platforms
  - wave_insitu.visualization : Interactive map generation
  - wave_insitu.utils : Common utility functions
"""

__version__ = "0.1.0"
__author__ = "Edouard Gauvrit"
__email__ = "edouard.gauvrit@ifremer.fr"

from wave_insitu.loaders import saildrone, ldl, kub, tc
from wave_insitu.visualization import map

__all__ = [
    "saildrone",
    "ldl",
    "kub",
    "tc",
    "map",
]
