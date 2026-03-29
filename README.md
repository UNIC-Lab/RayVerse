# RayVerse

**RayVerse** is a large-scale urban radio propagation dataset generated via ray tracing, containing multipath propagation data (`.propbin`) and building geometry (`.json`) for 700 urban map scenarios. Each scenario includes multiple transmitter positions with full multipath information (path type, angle of arrival/departure, delay, path loss, etc.).

The dataset supports research on:

- **Received Signal Strength (RSS)** prediction
- **Angular Power Spectrum (APS)** estimation
- **Power Delay Profile (PDP)** analysis
- **Antenna beam pattern** evaluation

## Download

The full dataset is available on Baidu Netdisk:

**Link:** https://pan.baidu.com/s/1Ze47tm5WohUUNBX8UUTQlg  **Password:** `vkm3`

## Repository Structure

```
RayVerse/
├── aps/                                    # APS visualization examples (PNG)
├── pdp/                                    # PDP visualization examples (PNG)
├── rss/                                    # RSS visualization examples (PNG)
└── src/
    ├── conversion/
    │   ├── propbin_to_aps_pdp.py           # Convert propbin to APS/PDP numpy arrays
    │   └── apply_sinc_beam.py              # Apply sinc antenna beam pattern to propbin
    ├── visualization/
    │   ├── visualize_rss.py                # Visualize a single RSS heatmap
    │   └── visualize_aps_pdp.py            # Visualize a single APS/PDP curve
    └── utils/
        ├── propbin_reader.py               # Read .propbin / .propbin.gz files
        ├── propbin_writer.py               # Write .propbin v2 files
        └── proj_geometry.py                # Geometric projection utilities
```

## Dataset Format

The downloaded dataset has the following top-level structure:

```
RayVerse_data/
├── buildings_complete/                         # Building geometry for all maps
│   ├── 0.json
│   ├── 1.json
│   └── ...
├── map_0/
│   └── special_points_propbin_{FREQ}/          # e.g. special_points_propbin_3.5GHz
│       ├── source_0.propbin.gz                 # Multipath propagation data for TX 0
│       ├── source_1.propbin.gz                 # Multipath propagation data for TX 1
│       └── ...
├── map_1/
│   └── special_points_propbin_{FREQ}/
│       └── ...
└── ...
```

- **`{FREQ}`** is the carrier frequency used in ray tracing, e.g. `3.5GHz`.
- Building geometry for map `N` is stored in `buildings_complete/N.json`.

The `.propbin` binary format stores per-path information including:

| Field | Description |
|-------|-------------|
| Path type | LoS, reflection, diffraction, etc. |
| AoD / AoA | Angle of departure / arrival (azimuth & elevation) |
| Delay | Propagation delay (ns) |
| Distance | Total path length (m) |
| Path loss | Per-path attenuation (dB) |

Use `src/utils/propbin_reader.py` to load `.propbin` files in Python.

## Quick Start

### Read a propbin file

```python
from src.utils.propbin_reader import load_propbin

data = load_propbin("map_0/special_points_propbin_3.5GHz/source_0.propbin.gz")
```

### Visualize RSS heatmap

```bash
python src/visualization/visualize_rss.py \
    map_0/special_points_propbin_3.5GHz/source_0.propbin.gz --map-id 0
```

### Convert to APS / PDP

```bash
python src/conversion/propbin_to_aps_pdp.py
```

Configuration (grid resolution, input/output paths) is set inside the script.

### Apply antenna beam pattern

```bash
python src/conversion/apply_sinc_beam.py \
    --input-root /path/to/propbin_dir \
    --output-root /path/to/beamed_output \
    --map-id-start 0 --map-id-end 99 \
    --tx-boresight-az 0.0 \
    --tx-boresight-el 0.0 \
    --az-mainlobe-width 30.0 \
    --el-mainlobe-width 30.0
```

### Visualize APS / PDP curves

```bash
python src/visualization/visualize_aps_pdp.py \
    --root /path/to/aps_pdp_dir \
    --name "aps_0_100_200_150_180"
```

## Visualization Examples

Output filenames follow the convention below:

| Type | Filename | Description |
|------|----------|-------------|
| RSS  | `{map_id}_{tx_x}_{tx_y}.png` | RSS heatmap for a given TX position |
| APS  | `aps_{map}_{src}_{tx_x}_{tx_y}_{rx}.png` | Angular power spectrum for a TX-RX pair |
| PDP  | `pdp_{map}_{src}_{tx_x}_{tx_y}_{rx}.png` | Power delay profile for a TX-RX pair |

### RSS Heatmap
<p align="center"><img src="rss/example.png" width="400"></p>

### Angular Power Spectrum (APS)
<p align="center"><img src="aps/example.png" width="400"></p>

### Power Delay Profile (PDP)
<p align="center"><img src="pdp/example.png" width="400"></p>

## License

This dataset is released under the CC BY 4.0 License.
