"""Assign distance-to-Polar-Front to each CPR segment.

Source:
    data/Polar_Front_weekly.nc
        Freeman & Lovenduski (2016) weekly Polar Front (PF) position derived
        from altimetry SSH gradients.
        Variable PFw[time, longitude] = latitude of the PF at each 0.25-deg
        longitude in each week. Coverage: 2002-06-08 .. 2014-02-22 (612
        weekly Saturday timestamps).

For each CPR segment we look up the PF latitude at the segment's longitude.
- If the segment date is inside the PF coverage window we use the actual
  weekly PF for the nearest Saturday.
- Otherwise we fall back to a **climatological** PF computed from the same
  product: mean PF latitude per (ISO week-of-year, longitude) across all
  available years (2002-2014).

Distance is the signed great-circle meridional distance from the sample to
the front along the sample's longitude:
    dist = R_earth * (sample_lat - pf_lat) * pi / 180
- positive : sample is NORTH of the front (equatorward)
- negative : sample is SOUTH of the front (poleward)

To keep weekly-based and climatology-based distances separated (they have
different provenance / accuracy), we write them to TWO numeric columns:
    dist_to_PF_weekly_km : filled only for samples within the PF coverage
                           window 2002-06-08 .. 2014-02-22 (~57% of CPR);
                           NaN otherwise.
    dist_to_PF_clim_km   : filled only for samples OUTSIDE that window
                           (~43%, using week-of-year climatology); NaN otherwise.
Thus each row has exactly one of the two filled (or neither if the row is bad).

Inputs/outputs:
- Reads  data/metadata/segment_metadata_with_env.csv (or segment_metadata.csv)
- Writes data/metadata/segment_metadata_with_env.csv (overwrites in place,
         adds two numeric columns: dist_to_PF_weekly_km, dist_to_PF_clim_km;
         removes any earlier dist_to_PF_km / dist_to_PF_source).

Run from project root:
    uv run --with xarray --with netCDF4 --with numpy --with pandas \\
        python scripts/assign_pf_distance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_WITH_ENV = PROJECT_ROOT / "data" / "metadata" / "segment_metadata_with_env.csv"
META_BARE = PROJECT_ROOT / "data" / "metadata" / "segment_metadata.csv"
PF_NC = PROJECT_ROOT / "data" / "Polar_Front_weekly.nc"

EARTH_R_KM = 6371.0088  # mean Earth radius


def load_pf(pf_nc: Path):
    """Open the Freeman & Lovenduski weekly PF and return:
      - pf_lat: ndarray (n_weeks, n_lon)
      - pf_dates: pandas DatetimeIndex of length n_weeks
      - pf_lons: ndarray (n_lon,) in [0, 360)
      - climatology: ndarray (53, n_lon) mean PF lat per ISO week-of-year
    """
    ds = xr.open_dataset(pf_nc, decode_times=False)
    pf_lat = ds["PFw"].values.astype(np.float64)               # (T, L)
    pf_lons = ds["longitude"].values.astype(np.float64)         # (L,)

    # The file's `time_stamp` is YYYYMMDD as float; use it as the canonical
    # date for each weekly slice (Saturday end-of-week per the file's docs).
    ts = ds["time_stamp"].values.astype(np.int64)               # (T,)
    pf_dates = pd.to_datetime(ts.astype(str), format="%Y%m%d")
    ds.close()

    # Climatology: mean over all years for each ISO week-of-year (1..53).
    iso_week = pf_dates.isocalendar().week.values.astype(int)
    clim = np.full((54, pf_lat.shape[1]), np.nan, dtype=np.float64)
    for wk in range(1, 54):
        mask = iso_week == wk
        if mask.any():
            clim[wk] = np.nanmean(pf_lat[mask, :], axis=0)
    # If any week has all-NaN (rare), back-fill with nearest valid week.
    valid = ~np.isnan(clim).all(axis=1)
    if not valid.all():
        # simple ffill/bfill along week axis
        idx = np.arange(54)
        valid_idx = idx[valid]
        for w in idx:
            if not valid[w] and valid_idx.size:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - w))]
                clim[w] = clim[nearest]

    return pf_lat, pf_dates, pf_lons, clim


def pf_lat_for_segments(
    seg_dates: np.ndarray,
    seg_lon360: np.ndarray,
    pf_lat: np.ndarray,
    pf_dates: pd.DatetimeIndex,
    pf_lons: np.ndarray,
    clim: np.ndarray,
):
    """For each segment, return (pf_lat_at_segment, source) arrays.

    source is 'weekly' if the segment date is within PF coverage and a
    nearest-week match exists, else 'climatology'.
    """
    n = seg_dates.size

    # nearest longitude index for every segment
    # pf_lons step is 0.25 starting at 0.125 -> idx = round((lon - 0.125)/0.25)
    lon_idx = np.round((seg_lon360 - pf_lons[0]) / (pf_lons[1] - pf_lons[0])).astype(int)
    lon_idx = np.clip(lon_idx, 0, pf_lons.size - 1)

    # ---- weekly lookup
    pf_t_ns = pf_dates.values.astype("datetime64[ns]").astype(np.int64)
    seg_t_ns = seg_dates.astype("datetime64[ns]").astype(np.int64)

    in_window = (seg_t_ns >= pf_t_ns.min()) & (seg_t_ns <= pf_t_ns.max())

    # For in-window segments, find nearest weekly index via searchsorted+adjust
    weekly_lat = np.full(n, np.nan, dtype=np.float64)
    if in_window.any():
        sub_t = seg_t_ns[in_window]
        right = np.searchsorted(pf_t_ns, sub_t, side="left")
        right = np.clip(right, 0, pf_t_ns.size - 1)
        left = np.clip(right - 1, 0, pf_t_ns.size - 1)
        # pick whichever (left or right) is closer in time
        choose_right = np.abs(pf_t_ns[right] - sub_t) <= np.abs(pf_t_ns[left] - sub_t)
        t_idx = np.where(choose_right, right, left)
        weekly_lat[in_window] = pf_lat[t_idx, lon_idx[in_window]]

    # ---- climatology fallback
    seg_pd = pd.DatetimeIndex(seg_dates)
    iso_week = seg_pd.isocalendar().week.values.astype(int)  # 1..53
    iso_week = np.clip(iso_week, 1, 53)
    clim_lat = clim[iso_week, lon_idx]                        # (n,)

    # combine
    out_lat = weekly_lat.copy()
    out_source = np.where(np.isfinite(weekly_lat), "weekly", "climatology")
    mask_fb = ~np.isfinite(weekly_lat)
    out_lat[mask_fb] = clim_lat[mask_fb]

    return out_lat, out_source


def signed_distance_km(sample_lat: np.ndarray, pf_lat: np.ndarray) -> np.ndarray:
    """Signed great-circle distance along the same longitude.

    Positive : sample north of front (sample_lat > pf_lat)
    Negative : sample south of front
    Units    : km
    """
    diff_deg = sample_lat - pf_lat
    return EARTH_R_KM * np.deg2rad(diff_deg)


def main() -> int:
    t_total = time.time()

    if not PF_NC.exists():
        print(f"ERROR: {PF_NC} not found", file=sys.stderr)
        return 2

    # Prefer the env-augmented file if it exists, else the bare metadata.
    if META_WITH_ENV.exists():
        in_path = META_WITH_ENV
    else:
        in_path = META_BARE
    print(f"Reading {in_path} ...")
    seg = pd.read_csv(in_path)
    n_in = len(seg)

    seg["Date_parsed"] = pd.to_datetime(seg["Date"], format="%d-%b-%Y",
                                        errors="coerce")
    bad = seg["Date_parsed"].isna() | seg[["Latitude", "Longitude"]].isna().any(axis=1)
    if bad.any():
        print(f"  WARNING: {int(bad.sum())} rows have bad date or lat/lon -> NaN.")

    print(f"Loading PF from {PF_NC} ...")
    pf_lat, pf_dates, pf_lons, clim = load_pf(PF_NC)
    print(f"  PF coverage: {pf_dates.min().date()} .. {pf_dates.max().date()} "
          f"({pf_dates.size} weekly slices, {pf_lons.size} longitudes)")
    print(f"  Climatology built: {(~np.isnan(clim).all(axis=1)).sum()} week-of-year slices valid")

    good = ~bad
    seg_dates = seg.loc[good, "Date_parsed"].values
    seg_lon = seg.loc[good, "Longitude"].astype(float).values
    seg_lat = seg.loc[good, "Latitude"].astype(float).values
    seg_lon360 = np.where(seg_lon < 0, seg_lon + 360.0, seg_lon)

    print(f"Looking up PF for {seg_dates.size} segments ...")
    t0 = time.time()
    pf_lat_seg, src = pf_lat_for_segments(seg_dates, seg_lon360,
                                          pf_lat, pf_dates, pf_lons, clim)
    print(f"  done in {time.time()-t0:.2f} s")

    dist_km = signed_distance_km(seg_lat, pf_lat_seg)

    # Drop any pre-existing PF columns from older runs so we end up clean.
    for old in ("dist_to_PF_km", "dist_to_PF_source"):
        if old in seg.columns:
            seg = seg.drop(columns=[old])

    # Split into two source-specific numeric columns; each row has exactly
    # one of them filled (the other is NaN).
    seg["dist_to_PF_weekly_km"] = np.nan
    seg["dist_to_PF_clim_km"] = np.nan
    good_idx = seg.index[good]
    mask_weekly = (src == "weekly")
    mask_clim   = (src == "climatology")
    seg.loc[good_idx[mask_weekly], "dist_to_PF_weekly_km"] = dist_km[mask_weekly]
    seg.loc[good_idx[mask_clim],   "dist_to_PF_clim_km"]   = dist_km[mask_clim]

    # Clean up helper column
    seg = seg.drop(columns=["Date_parsed"])

    out_path = META_WITH_ENV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_path, index=False)

    # ---------------- summary ----------------
    n_weekly = int(seg["dist_to_PF_weekly_km"].notna().sum())
    n_clim   = int(seg["dist_to_PF_clim_km"].notna().sum())
    n_miss   = int(len(seg) - n_weekly - n_clim)
    print("\n===== SUMMARY =====")
    print(f"  rows written              : {len(seg)} -> {out_path}")
    print(f"  dist_to_PF_weekly_km set  : {n_weekly:5d}  ({n_weekly/len(seg)*100:5.1f}%)")
    print(f"  dist_to_PF_clim_km set    : {n_clim:5d}  ({n_clim/len(seg)*100:5.1f}%)")
    print(f"  neither set (bad row)     : {n_miss:5d}  ({n_miss/len(seg)*100:5.1f}%)")
    print()
    print("dist_to_PF_weekly_km stats:")
    print(seg["dist_to_PF_weekly_km"].describe())
    print()
    print("dist_to_PF_clim_km stats:")
    print(seg["dist_to_PF_clim_km"].describe())
    print()
    print("Sign convention: positive = sample north of PF (equatorward),"
          " negative = south of PF (poleward).")
    print(f"\nTotal wall time: {time.time() - t_total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
