#!/usr/bin/env python3
"""
Download SST and sea ice concentration from ERDDAP/HadISST,
match to CPR segments by location and month/year,
then re-aggregate into H3 hexagons.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import urllib.request
import gzip
import shutil

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
ENV_DIR = DATA / "environmental"
ENV_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def download_hadisst_sst():
    """Download HadISST SST from ERDDAP (Southern Ocean subset, 1991-2023)."""
    out = ENV_DIR / "hadisst_sst.nc"
    if out.exists():
        print(f"  SST already downloaded: {out}")
        return out

    print("  Downloading HadISST SST from ERDDAP (1991-2023, south of 35S)...")
    url = (
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdHadISST.nc?"
        "sst[(1991-01-16T00:00:00Z):1:(2023-12-16T00:00:00Z)]"
        "[(-89.5):1:(-35.5)]"
        "[(-179.5):1:(179.5)]"
    )
    urllib.request.urlretrieve(url, out)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  → {out.name}: {size_mb:.1f} MB")
    return out


def download_hadisst_ice():
    """Download HadISST sea ice concentration from Met Office (full file, ~12MB)."""
    out = ENV_DIR / "hadisst_ice.nc"
    if out.exists():
        print(f"  Sea ice already downloaded: {out}")
        return out

    print("  Downloading HadISST sea ice from Met Office...")
    gz_path = ENV_DIR / "HadISST_ice.nc.gz"
    url = "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_ice.nc.gz"
    urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  → {out.name}: {size_mb:.1f} MB")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATCH TO CPR SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────

def extract_monthly_values(ds, var_name, lats, lons, years, months):
    """
    For each segment (lat, lon, year, month), extract the nearest grid value.
    Returns a numpy array of values aligned with the input arrays.
    """
    n = len(lats)
    values = np.full(n, np.nan)

    time_index = pd.DatetimeIndex(ds.time.values)
    ym_to_tidx = {}
    for i, t in enumerate(time_index):
        ym_to_tidx[(t.year, t.month)] = i

    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values

    for i in range(n):
        key = (int(years[i]), int(months[i]))
        if key not in ym_to_tidx:
            continue

        tidx = ym_to_tidx[key]
        lat_idx = np.argmin(np.abs(lat_vals - lats[i]))
        lon_idx = np.argmin(np.abs(lon_vals - lons[i]))

        val = float(ds[var_name].values[tidx, lat_idx, lon_idx])
        if val > -100 and val < 100:
            values[i] = val

    return values


def main():
    print("=== Downloading environmental data ===\n")
    sst_path = download_hadisst_sst()
    ice_path = download_hadisst_ice()

    print("\n=== Loading CPR segment metadata ===")
    meta = pd.read_csv(DATA / "metadata" / "segment_metadata.csv")
    meta["Month_num"] = pd.to_datetime(meta["Month"], format="%B").dt.month
    print(f"  {len(meta)} segments")

    print("\n=== Extracting SST at segment locations ===")
    ds_sst = xr.open_dataset(sst_path)
    meta["hadisst_sst"] = extract_monthly_values(
        ds_sst, "sst",
        meta["Latitude"].values, meta["Longitude"].values,
        meta["Year"].values, meta["Month_num"].values,
    )
    valid_sst = meta["hadisst_sst"].notna().sum()
    print(f"  Matched {valid_sst}/{len(meta)} segments ({100*valid_sst/len(meta):.1f}%)")
    ds_sst.close()

    print("\n=== Extracting sea ice concentration at segment locations ===")
    ds_ice = xr.open_dataset(ice_path)
    meta["hadisst_ice"] = extract_monthly_values(
        ds_ice, "sic",
        meta["Latitude"].values, meta["Longitude"].values,
        meta["Year"].values, meta["Month_num"].values,
    )
    valid_ice = meta["hadisst_ice"].notna().sum()
    print(f"  Matched {valid_ice}/{len(meta)} segments ({100*valid_ice/len(meta):.1f}%)")
    ds_ice.close()

    out_path = DATA / "metadata" / "segment_env_data.csv"
    env_cols = meta[["Segment_ID", "hadisst_sst", "hadisst_ice"]].copy()
    env_cols.to_csv(out_path, index=False)
    print(f"\n  → {out_path.name}: {len(env_cols)} rows")

    print(f"\n  SST range: {meta['hadisst_sst'].min():.1f} to {meta['hadisst_sst'].max():.1f} °C")
    print(f"  Ice range: {meta['hadisst_ice'].min():.2f} to {meta['hadisst_ice'].max():.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
