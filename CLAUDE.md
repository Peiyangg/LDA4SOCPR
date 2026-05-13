# CPR Data Analysis — Project Instructions

## D3 Visualization Workflow

The interactive D3 visualization lives at `figures/index.html`. The user views it through a Marimo notebook (`notebooks/CPR_D3_Explorer.py`) which serves the `figures/` directory via a local HTTP server and embeds it in an iframe.

**After updating `figures/index.html` or any JS data files, do NOT run `open figures/index.html`.** The user already has the Marimo notebook open — just tell them to refresh the iframe (reload the Marimo page or re-run the iframe cell).

## Key Files

- `figures/index.html` — D3 + Canvas interactive map (Southern Ocean CPR explorer)
- `figures/*.js` — data files loaded by index.html:
  - `cpr_data.js` — per-segment records (lat, lon, year, month, segLen, shannon, ship, tow, seg#, date, time, groups[], season)
  - `hex_res2.js`, `hex_res3.js` — H3 hex-binned data per austral season
  - `fronts_data.js` — Orsi 1995 Southern Ocean fronts GeoJSON
  - `world_data.js` — world-atlas land polygons for basemap
  - `nmf_data.js` — NMF topic compositions (top-5 species per topic, grouped by season/tow)
  - `nmf_weights.js` — per-segment NMF weights keyed by segment index (for map colouring)
  - `pf_weekly_data.js` — weekly Polar Front latitude by date (Freeman & Lovenduski 2016, Nov 2002–Mar 2014, austral summer only)
- `notebooks/preprocess_hex_viz.py` — preprocesses CPR data into H3 hex bins (per austral season) and fronts GeoJSON
- `notebooks/preprocess_pf_weekly.py` — preprocesses weekly PF NetCDF into per-season min/max bands, outputs pf_weekly_data.js
- `notebooks/preprocess_nmf_viz.py` — reads per-tow NMF results, outputs nmf_data.js and nmf_weights.js
- `notebooks/preprocess_viz.py` — raw segment data (with season field) for D3, outputs cpr_data.js and world_data.js
- `notebooks/download_env_data.py` — downloads HadISST SST and sea ice, matches to CPR segments
- `notebooks/CPR_D3_Explorer.py` — Marimo notebook that embeds the D3 viz via iframe
- `data/nmf_per_tow/` — per-tow NMF results (W/H matrices, meta.json, summary.json)
- `data/hex_features_res{2,3}.csv` — 272 raw species abundances per hex-season
- `data/hex_metadata_res{2,3}.csv` — environmental/effort metadata per hex-season
- `data/hex_effort_res{2,3}.csv` — pivot table: hex × season → sample count

## Data Pipeline

1. `download_env_data.py` → downloads SST/ice NetCDF, extracts per-segment values
2. `preprocess_hex_viz.py` → H3 binning at res 2 & 3 by austral season, outputs JS + CSV files
3. `preprocess_viz.py` → raw segment data (with season field) for D3
4. `preprocess_nmf_viz.py` → NMF topic data + per-segment weights for D3

## Current State of the D3 Visualization

The map (`figures/index.html`) supports two main view modes:

### Raw Segment Mode
- Segments are coloured by selected metric (Shannon diversity, species group abundance, etc.)
- Segment circle size is scaled by **total abundance** (log1p transform, sqrt scale, range [1.5, 7] px)
- Hover highlights only the single hovered segment (not the whole tow)
- Tooltip shows: ship, tow, segment, season, date, lat/lon, month, length, Shannon H', total abundance, and selected group count

### NMF Topic Mode
- Two-level dropdown: first select a **tow group**, then a **topic** within that tow
- When a tow is selected, non-matching segments are dimmed (muted gray at colorLUT[0])
- Matching segments are coloured by topic weight (0–1)
- Legend panel (bottom-right) shows only the selected tow's topics with species bar charts (hidden when no tow selected)
- NMF dropdown only rebuilds when the season changes (tracked via `_lastNmfDropdownSeason`)

### Hex Mode
- H3 hexagonal binning at resolution 2 or 3, aggregated per austral season
- Hexes coloured by selected metric

### General
- All temporal filtering uses **austral summer seasons** (Nov–Mar, labelled like "2000-01"), not calendar years
- LDA has been fully removed and replaced by NMF
- Segment record layout: `[lat, lon, year, monthIdx, segLen, shannon, ship, towNo, segNo, date, time, groups[], season]`
- `_origIdx` on each segment maps back to the index in `cpr_data.js` for NMF weight lookup via `window.NMF_WEIGHTS[_origIdx]`

## Virtual Environment

Uses `uv` for package management. Python is at `.venv/bin/python`.
