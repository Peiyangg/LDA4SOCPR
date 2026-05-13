"""Assign daily SST and sea-ice concentration (IC) to each CPR segment.

Inputs:
  - data/metadata/segment_metadata.csv
  - /Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/
      NatureScience/POLARIS/SST/sst.day.mean.YYYY.nc
  - /Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/
      NatureScience/POLARIS/SI/icec.day.mean.YYYY.nc

Output:
  - data/metadata/segment_metadata_with_env.csv  (adds two columns: SST, IC)

How it works:
  For each calendar year that appears in the segment metadata, we open the
  matching yearly SST + ICEC NetCDFs and use xarray vectorised `.sel(...,
  method="nearest")` to extract values for every segment in that year in a
  single call.  No data is fabricated; segments that fall on dates / pixels
  with missing values get NaN.

Run with uv from project root:
  uv run --with xarray --with netCDF4 --with numpy --with pandas \
      python scripts/assign_sst_sic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEG_META_PATH = PROJECT_ROOT / "data" / "metadata" / "segment_metadata.csv"
OUT_PATH = PROJECT_ROOT / "data" / "metadata" / "segment_metadata_with_env.csv"

POLARIS_ROOT = (
    PROJECT_ROOT.parent  # .../POLARIS/
)
SST_DIR = POLARIS_ROOT / "SST"
SI_DIR = POLARIS_ROOT / "SI"


def sst_path(year: int) -> Path:
    return SST_DIR / f"sst.day.mean.{year}.nc"


def icec_path(year: int) -> Path:
    return SI_DIR / f"icec.day.mean.{year}.nc"


def main() -> int:
    t_total = time.time()

    if not SEG_META_PATH.exists():
        print(f"ERROR: not found: {SEG_META_PATH}", file=sys.stderr)
        return 2

    print(f"Reading {SEG_META_PATH} ...")
    seg = pd.read_csv(SEG_META_PATH)
    n_in = len(seg)

    # Parse date column. Source format example: "12-Jan-1991".
    seg["Date_parsed"] = pd.to_datetime(seg["Date"], format="%d-%b-%Y", errors="coerce")
    n_bad_date = int(seg["Date_parsed"].isna().sum())
    if n_bad_date:
        print(f"  WARNING: {n_bad_date} rows have an unparseable Date; they will get NaN.")
    n_bad_loc = int(seg[["Latitude", "Longitude"]].isna().any(axis=1).sum())
    if n_bad_loc:
        print(f"  WARNING: {n_bad_loc} rows have missing lat/lon; they will get NaN.")

    # Convert longitude to 0..360 to match OISST grid.
    seg["lon360"] = np.where(seg["Longitude"] < 0,
                             seg["Longitude"] + 360,
                             seg["Longitude"])

    # Result columns, default NaN
    seg["SST"] = np.nan
    seg["IC"] = np.nan

    # Group by year and process each year's NetCDFs once.
    valid = seg["Date_parsed"].notna() & seg[["Latitude", "Longitude"]].notna().all(axis=1)
    years = sorted(seg.loc[valid, "Date_parsed"].dt.year.unique())
    print(f"\n{n_in} segments total; "
          f"{int(valid.sum())} usable; "
          f"years to process: {years[0]}..{years[-1]} (n={len(years)})\n")

    missing_files: list[str] = []
    per_year_stats = []

    for year in years:
        sst_p = sst_path(int(year))
        icec_p = icec_path(int(year))

        if not sst_p.exists():
            print(f"  [{year}] MISSING {sst_p.name} -- skipping SST for this year")
            missing_files.append(sst_p.name)
        if not icec_p.exists():
            print(f"  [{year}] MISSING {icec_p.name} -- skipping IC for this year")
            missing_files.append(icec_p.name)

        mask = valid & (seg["Date_parsed"].dt.year == year)
        n = int(mask.sum())
        if n == 0:
            continue

        idx = seg.index[mask]
        times = seg.loc[mask, "Date_parsed"].values
        lats = seg.loc[mask, "Latitude"].astype(float).values
        lons = seg.loc[mask, "lon360"].astype(float).values

        t0 = time.time()

        if sst_p.exists():
            with xr.open_dataset(sst_p) as ds_sst:
                seg_dim = "seg"
                seg_t = xr.DataArray(times, dims=seg_dim)
                seg_lat = xr.DataArray(lats, dims=seg_dim)
                seg_lon = xr.DataArray(lons, dims=seg_dim)
                sst_vals = (
                    ds_sst["sst"]
                    .sel(time=seg_t, lat=seg_lat, lon=seg_lon, method="nearest")
                    .values
                )
            seg.loc[idx, "SST"] = sst_vals
            sst_ok = int(np.isfinite(sst_vals).sum())
        else:
            sst_ok = 0

        if icec_p.exists():
            with xr.open_dataset(icec_p) as ds_ic:
                seg_dim = "seg"
                seg_t = xr.DataArray(times, dims=seg_dim)
                seg_lat = xr.DataArray(lats, dims=seg_dim)
                seg_lon = xr.DataArray(lons, dims=seg_dim)
                ic_vals = (
                    ds_ic["icec"]
                    .sel(time=seg_t, lat=seg_lat, lon=seg_lon, method="nearest")
                    .values
                )
            seg.loc[idx, "IC"] = ic_vals
            ic_finite = int(np.isfinite(ic_vals).sum())
        else:
            ic_finite = 0

        dt = time.time() - t0
        per_year_stats.append(
            {
                "year": int(year),
                "n_segments": n,
                "sst_finite": sst_ok,
                "ic_finite": ic_finite,
                "seconds": round(dt, 2),
            }
        )
        print(
            f"  [{year}] {n:5d} segs   "
            f"SST finite {sst_ok:5d}   IC finite {ic_finite:5d}   "
            f"{dt:5.2f}s"
        )

    # Drop the helper column so the output stays clean.
    seg = seg.drop(columns=["Date_parsed", "lon360"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seg.to_csv(OUT_PATH, index=False)

    # ---------------- final report ----------------
    n_sst = int(seg["SST"].notna().sum())
    n_ic = int(seg["IC"].notna().sum())
    print("\n===== SUMMARY =====")
    print(f"  rows written            : {len(seg)} -> {OUT_PATH}")
    print(f"  SST non-NaN             : {n_sst}/{len(seg)} ({n_sst/len(seg)*100:.1f}%)")
    print(f"  IC  non-NaN             : {n_ic}/{len(seg)}  "
          f"({n_ic/len(seg)*100:.1f}%) "
          f"-- IC=NaN means <15% ice / land / open ocean")
    if missing_files:
        print(f"  yearly files missing    : {len(missing_files)}")
        for f in missing_files:
            print(f"    - {f}")
    print(f"\nTotal wall time: {time.time() - t_total:.1f} s")

    # Print yearly stats compactly
    if per_year_stats:
        print("\nPer-year detail:")
        df = pd.DataFrame(per_year_stats)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
