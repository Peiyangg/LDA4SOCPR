# CPR Data Analysis — Project Instructions

## D3 Visualization Workflow

The interactive D3 visualization lives at `figures/index.html`. The user views it through a Marimo notebook (`notebooks/CPR_D3_Explorer.py`) which serves the `figures/` directory via a local HTTP server and embeds it in an iframe.

**After updating `figures/index.html` or any JS data files, do NOT run `open figures/index.html`.** The user already has the Marimo notebook open — just tell them to refresh the iframe (reload the Marimo page or re-run the iframe cell).

## Key Files

- `figures/index.html` — D3 + Canvas interactive map (Southern Ocean CPR explorer)
- `figures/*.js` — data files loaded by index.html (hex_res2/3.js, fronts_data.js, cpr_data.js, world_data.js, lda_k*.js)
- `notebooks/preprocess_hex_viz.py` — preprocesses CPR data into H3 hex bins (per year) and fronts GeoJSON
- `notebooks/download_env_data.py` — downloads HadISST SST and sea ice, matches to CPR segments
- `notebooks/CPR_D3_Explorer.py` — Marimo notebook that embeds the D3 viz via iframe
- `data/hex_features_res{2,3}.csv` — 272 raw species abundances per hex-year (for future LDA)
- `data/hex_metadata_res{2,3}.csv` — environmental/effort metadata per hex-year
- `data/hex_effort_res{2,3}.csv` — pivot table: hex × year → sample count

## Data Pipeline

1. `download_env_data.py` → downloads SST/ice NetCDF, extracts per-segment values
2. `preprocess_hex_viz.py` → H3 binning at res 2 & 3, outputs JS + CSV files
3. `preprocess_viz.py` → raw segment data + LDA topic distributions for D3

## Virtual Environment

Uses `uv` for package management. Python is at `.venv/bin/python`.
