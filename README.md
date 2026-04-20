# wave-insitu

Tools for processing and visualizing in-situ ocean wave observations from multiple platforms.

## Overview

`wave-insitu` is a Python toolkit for loading, normalizing, and visualizing wave observations from:
- **Saildrone**: Autonomous sailboats with wave/wind sensors (multiple providers: CMEMS, NOAA, PIMEP)
- **LDL DWSD**: Lagrangian Drifter Laboratory Directional Wave Spectral Drifters
- **KU-Buoys**: Kyoto University SPOT buoys  

## Installation

### With pip

```bash
git clone https://github.com/umr-lops/wave-insitu.git
cd wave-insitu
pip install -e .
```

### With micromamba (recommended)

```bash
git clone https://github.com/umr-lops/wave-insitu.git
cd wave-insitu
micromamba env create -f environment.yml
micromamba activate wave-insitu
```

### With conda

```bash
git clone https://github.com/umr-lops/wave-insitu.git
cd wave-insitu
conda env create -f environment.yml
conda activate wave-insitu
```

## Scripts

### build_insitu_catalog.py

Merges observations from all platforms into a single CSV catalog.

**Quick Start:**
```bash
python scripts/build_insitu_catalog.py --verbose
```
Uses default config from `config/data_dirs.yaml`. Outputs: `wave_obs_catalog.csv` with ~1M+ observations.

**What it does:**
1. Scans Saildrone directories (configurable providers: CMEMS, NOAA, PIMEP)
2. Loads LDL DWSD drifter data
3. Loads KU-Buoys data
4. Optionally filters by wave/wind presence
5. Merges all into unified DataFrame
6. Saves to CSV

**Options:**

Default (uses `config/data_dirs.yaml`):
```bash
python scripts/build_insitu_catalog.py --verbose
```

Custom config path:
```bash
python scripts/build_insitu_catalog.py \
    --data-dirs /path/to/data_dirs.yaml \
    --verbose
```

Override paths individually:
```bash
python scripts/build_insitu_catalog.py \
    --saildrone cmems:/path/to/cmems noaa:/path/to/noaa pimep:/path/to/pimep \
    --ldl-dir /path/to/ldl \
    --kub-dir /path/to/kub \
    --query-condition wave \
    --output wave_obs_catalog.csv \
    --verbose
```

**Output Format:**

All rows have these **metadata columns**:
| Column | Type | Values |
|--------|------|--------|
| `time` | datetime | UTC timestamp |
| `latitude` | float | degrees (-90 to 90) |
| `longitude` | float | degrees (-180 to 180) |
| `platform_type` | str | `saildrone` \| `dwsd` \| `spotter` |
| `platform_id` | str | Unique platform ID |
| `provider` | str | Data provider organization |
| `name` | str | Track/trajectory name |
| `source_file` | str | Original file path |
| `tc_name` | str | Associated tropical cyclone |

Plus **measured variables** (platform-dependent):
- `significant_wave_height` : Wave height (m)
- `dominant_wave_period` : Wave period (s)
- `wind_speed` : Wind speed (m/s)
- `wind_direction` : Wind direction (0-360°)
- And others...

**Configuration File:**

`config/data_dirs.yaml` - Centralized data source paths:
```yaml
data_sources:
  saildrone:
    cmems: "/home/ref-copernicus-insitu/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/cmems_obs-ins_glo_phybgcwav_mynrt_na_irr/history/SD"
    noaa: "/home/datawork-cersat-public/provider/noaa/insitu/saildrone"
    pimep: "/home/datawork-cersat-public/project/pimep/data/saildrone"
  
  ldl:
    path: "/scale/user/egauvrit/data/insitu/DWSD"
  
  kub:
    path: "/scale/user/egauvrit/data/insitu/KUB/SWH"
```

`config/mapping.yaml` - Variable name aliases for all platforms:
```yaml
variables:
  significant_wave_height:
    - "Hs"
    - "significant_wave_height"
    - "swh"
  wind_speed:
    - "wind"
    - "wind_speed"
    - "ws"
```

**Python API:**
```python
from scripts.build_insitu_catalog import build_insitu_catalog

catalog = build_insitu_catalog(
    saildrone_dirs={"cmems": "/path", "noaa": "/path", "pimep": "/path"},
    ldl_dir="/path/to/ldl",
    kub_dir="/path/to/kub",
    mapping_path="config/mapping.yaml",
    query_condition="wave",
    output_path="wave_obs_catalog.csv",
    verbose=True
)
```

### build_map_from_catalog.py

Generates interactive Folium map from catalog CSV.

**Quick Start:**
```bash
python scripts/build_map_from_catalog.py
```
Outputs: `insitu_map.html` - interactive map with TC overlays.

**What it does:**
1. Loads merged catalog CSV
2. Separates observations by platform type
3. Loads tropical cyclone tracks for the period
4. Builds interactive Folium map with:
   - Colored trajectories per platform
   - TC tracks colored by wind speed
   - Clickable measurement points
   - Layer controls for toggling data

**Options:**
```bash
python scripts/build_map_from_catalog.py \
    --catalog wave_obs_catalog.csv \
    --output insitu_map.html \
    --wind-colormap config/wind_faozi.cpt \
    --ws-vmin 0 --ws-vmax 150 \
    --max-points 1000
```

**Python API:**
```python
from scripts.build_map_from_catalog import build_map_from_catalog

build_map_from_catalog(
    catalog_path="wave_obs_catalog.csv",
    output_path="insitu_map.html",
    wind_colormap_path="config/wind_faozi.cpt"
)
```

## Project Structure

```
wave-insitu/
├── README.md                          # This file
├── LICENSE
├── pyproject.toml
│
├── wave_insitu/                        # Main package
│   ├── utils.py                        # Common utilities (load_mapping, build_reverse_lookup)
│   ├── loaders/
│   │   ├── saildrone.py               # Saildrone loader
│   │   ├── ldl.py                     # LDL DWSD loader
│   │   ├── kub.py                     # KU-Buoys loader
│   │   └── tc.py                      # Tropical cyclone tracks (CyclObs API)
│   └── visualization/
│       └── map.py                     # Folium map builder
│
├── scripts/
│   ├── build_insitu_catalog.py        # Build merged catalog
│   └── build_map_from_catalog.py      # Generate interactive map
│
└── config/
    ├── data_dirs.yaml                 # Centralized data source paths (recommended)
    ├── saildrone_dirs.yaml            # Saildrone provider paths (legacy)
    ├── mapping.yaml                   # Variable name aliases
    └── wind_faozi.cpt                 # Wind speed colormap
```

## Python API (Advanced)

### Load Individual Platforms

```python
from wave_insitu.loaders import saildrone, ldl, kub
from wave_insitu.utils import load_mapping

mapping = load_mapping("config/mapping.yaml")

# Saildrone (convenient multi-provider function)
sd_df = saildrone.load_saildrone_from_dirs(
    sddirs={"cmems": "/path/to/cmems", "noaa": "/path/to/noaa"},
    mapping=mapping,
    query_condition="wave and wind",
    verbose=True
)

# LDL DWSD
ldl_files = ldl.get_ldl_files("/path/to/ldl/data")
ldl_df = ldl.load_ldl_catalog(ldl_files)

# KU-Buoys
kub_files = list(Path("/path/to/kub").rglob("*.nc"))
kub_df = kub.load_kub_catalog(kub_files, mapping)
```

### Query & Filter

```python
import pandas as pd

# Load merged catalog
catalog = pd.read_csv("wave_obs_catalog.csv")

# By platform type
sd_only = catalog.query("platform_type == 'saildrone'")

# By provider
cmems = catalog.query("provider == 'Saildrone' and name.str.contains('1000')")

# By tropical cyclone
milton = catalog.query("tc_name == 'MILTON'")

# With wave data only
has_waves = catalog[catalog['significant_wave_height'].notna()]
```

### Build Custom Map

```python
from wave_insitu.loaders import tc
from wave_insitu.visualization import map as ism

# Load TC tracks
tc_df = tc.load_tc_tracks(
    min_date="2021-06-01",
    max_date="2021-09-30"
)

# Separate by platform
sd_df = catalog.query("platform_type == 'saildrone'")
ldl_df = catalog.query("platform_type == 'dwsd'")
kub_df = catalog.query("platform_type == 'spotter'")

# Build map
m = ism.build_insitu_map(
    saildrone_df=sd_df,
    ldl_df=ldl_df,
    tc_df=tc_df,
    kub_df=kub_df,
    cpt_path="config/wind_faozi.cpt",
    output_path="custom_map.html"
)
```

## Troubleshooting

**Q: Build is slow on first run**  
A: Scanning Saildrone directories is I/O intensive. Subsequent runs with cached CSV are fast.

**Q: Map file is huge (>100 MB)**  
A: Use `--max-points 500` in map builder to reduce trajectory detail.

**Q: Missing data directories**  
A: Check that `config/saildrone_dirs.yaml` paths exist on your system.

**Q: ModuleNotFoundError**  
A: Ensure you installed with `pip install -e .` and have activated the correct environment.

## Authors

- **Edouard Gauvrit** - edouard.gauvrit@ifremer.fr (UMR-LOPS, Ifremer)

## License

MIT License - see LICENSE file
