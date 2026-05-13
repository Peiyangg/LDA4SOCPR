#!/usr/bin/env python3
"""
Preprocess Freeman & Lovenduski (2016) weekly Polar Front data
into per-week PF latitude lines for D3 visualization.

Input:  data/Polar_Front_weekly.nc
Output: figures/pf_weekly_data.js — window.PF_WEEKLY = {
          lons: [...],
          weeks: {"2003-01-18": [...latitudes...], ...}
        }
        Only austral summer weeks (Nov–Mar) are included.
"""
import numpy as np
import pandas as pd
import xarray as xr
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGURES = BASE / "figures"

ds = xr.open_dataset(DATA / "Polar_Front_weekly.nc", decode_times=False)

dates = pd.to_datetime(ds["time_stamp"].values.astype(int).astype(str))
lons_360 = ds["longitude"].values
pf_lat = ds["PFw"].values  # (612, 1440)

# Convert longitudes from 0–360 to -180–180
lons_180 = np.where(lons_360 > 180, lons_360 - 360, lons_360)
sort_idx = np.argsort(lons_180)
lons_180 = lons_180[sort_idx]
pf_lat = pf_lat[:, sort_idx]

# Subsample longitudes (every 4th → 360 points)
step = 4
lon_idx = np.arange(0, len(lons_180), step)
lons_sub = np.round(lons_180[lon_idx], 2).tolist()
pf_sub = pf_lat[:, lon_idx]

# Filter to austral summer months only (Nov–Mar)
summer_mask = dates.month.isin([11, 12, 1, 2, 3])
summer_dates = dates[summer_mask]
summer_pf = pf_sub[summer_mask]

weeks = {}
for i, dt in enumerate(summer_dates):
    key = dt.strftime("%Y-%m-%d")
    lats = np.round(summer_pf[i], 3)
    weeks[key] = [None if np.isnan(v) else float(v) for v in lats]

result = {"lons": lons_sub, "weeks": weeks}

js = "window.PF_WEEKLY=" + json.dumps(result, separators=(",", ":")) + ";\n"
out = FIGURES / "pf_weekly_data.js"
out.write_text(js)
print(f"Wrote {out} ({len(weeks)} weeks, {out.stat().st_size / 1024:.0f} KB)")
print(f"Date range: {min(weeks)} to {max(weeks)}")
