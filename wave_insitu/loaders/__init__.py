"""
Data loaders for various in-situ ocean observation platforms.

Author: Edouard Gauvrit (edouard.gauvrit@ifremer.fr)

Supported platforms:
  - Saildrone : Autonomous sailboats with wave and wind sensors
  - LDL DWSD  : Lagrangian Drifter Laboratory Directional Wave Spectral Drifters
  - KU-Buoys  : Kyoto University SPOT buoys
  - TC tracks : Tropical cyclone track data
"""

from . import saildrone, ldl, kub, tc

__all__ = ["saildrone", "ldl", "kub", "tc"]
