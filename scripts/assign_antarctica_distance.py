"""Assign great-circle distance from each CPR segment to the nearest piece of
Antarctic coastline.

Coastline source:
    Natural Earth 50 m physical land shapefile (via cartopy). We keep every
    polygon whose centroid is south of -55 deg (= main continent + ice
    shelves + surrounding islands like the Antarctic Peninsula tip and the
    sub-Antarctic islands clinging to the continent).  Vertex spacing is
    ~5-10 km; we densify each polygon edge to <= 5 km so the nearest-vertex
    distance is a tight upper bound on the true point-to-polyline distance.

Method:
    All coastline vertices are placed on the unit sphere (x, y, z). We build
    a cKDTree on those points and query the nearest one for each CPR sample.
    Chord distance is converted back to great-circle distance:
        d_gc = 2 * R * arcsin(chord / 2)

Inputs/outputs:
- Reads  data/metadata/segment_metadata_with_env.csv (or segment_metadata.csv)
- Writes data/metadata/segment_metadata_with_env.csv  (overwrites; adds one
         column: dist_to_Antarctica_km)

Run from project root:
    uv run --with cartopy --with shapely --with numpy --with pandas --with scipy \\
        python scripts/assign_antarctica_distance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import cartopy.io.shapereader as shpreader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_WITH_ENV = PROJECT_ROOT / "data" / "metadata" / "segment_metadata_with_env.csv"
META_BARE = PROJECT_ROOT / "data" / "metadata" / "segment_metadata.csv"

EARTH_R_KM = 6371.0088
MAX_VERTEX_SPACING_KM = 5.0       # densify coastline edges to <= this length
CENTROID_LAT_CUTOFF = -55.0       # polygons with centroid < this are "Antarctic"


def lonlat_to_xyz(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg))
    lat = np.deg2rad(np.asarray(lat_deg))
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def densify(lons, lats, max_km=MAX_VERTEX_SPACING_KM):
    """Insert extra points along each line segment so successive points are
    at most `max_km` apart (rough great-circle distance estimate)."""
    out_lons = [lons[0]]
    out_lats = [lats[0]]
    for i in range(1, len(lons)):
        lon0, lat0 = lons[i - 1], lats[i - 1]
        lon1, lat1 = lons[i], lats[i]
        # rough great-circle distance via haversine
        rlat0, rlat1 = np.deg2rad(lat0), np.deg2rad(lat1)
        rdlat = rlat1 - rlat0
        rdlon = np.deg2rad(lon1 - lon0)
        a = np.sin(rdlat / 2) ** 2 + np.cos(rlat0) * np.cos(rlat1) * np.sin(rdlon / 2) ** 2
        d_km = 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))
        n_extra = int(np.ceil(d_km / max_km)) - 1
        if n_extra > 0:
            for k in range(1, n_extra + 1):
                f = k / (n_extra + 1)
                out_lons.append(lon0 + f * (lon1 - lon0))
                out_lats.append(lat0 + f * (lat1 - lat0))
        out_lons.append(lon1)
        out_lats.append(lat1)
    return np.asarray(out_lons), np.asarray(out_lats)


def collect_antarctic_coastline():
    """Return (lons, lats) numpy arrays of all Antarctic coastline vertices,
    densified to <= MAX_VERTEX_SPACING_KM between consecutive points."""
    shp = shpreader.natural_earth(
        resolution="50m", category="physical", name="land"
    )
    reader = shpreader.Reader(shp)

    all_lons = []
    all_lats = []
    n_poly = 0
    for geom in reader.geometries():
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue
        for poly in polys:
            if poly.centroid.y >= CENTROID_LAT_CUTOFF:
                continue
            lons, lats = poly.exterior.coords.xy
            lons = np.asarray(lons, dtype=float)
            lats = np.asarray(lats, dtype=float)
            d_lons, d_lats = densify(lons, lats)
            all_lons.append(d_lons)
            all_lats.append(d_lats)
            n_poly += 1

    lons = np.concatenate(all_lons)
    lats = np.concatenate(all_lats)
    return lons, lats, n_poly


def main() -> int:
    t_total = time.time()

    in_path = META_WITH_ENV if META_WITH_ENV.exists() else META_BARE
    print(f"Reading {in_path} ...")
    seg = pd.read_csv(in_path)
    n_in = len(seg)

    bad_loc = seg[["Latitude", "Longitude"]].isna().any(axis=1)
    if bad_loc.any():
        print(f"  WARNING: {int(bad_loc.sum())} rows have bad lat/lon -> NaN.")

    print("Loading Natural Earth 50m Antarctic coastline ...")
    t0 = time.time()
    coast_lons, coast_lats, n_poly = collect_antarctic_coastline()
    print(f"  {n_poly} polygons, {len(coast_lons)} vertices (after densify), "
          f"loaded in {time.time()-t0:.2f} s")
    print(f"  coast lat range: {coast_lats.min():.2f} .. {coast_lats.max():.2f}")
    print(f"  coast lon range: {coast_lons.min():.2f} .. {coast_lons.max():.2f}")

    print("Building KDTree on unit sphere ...")
    t0 = time.time()
    coast_xyz = lonlat_to_xyz(coast_lons, coast_lats)
    tree = cKDTree(coast_xyz)
    print(f"  done in {time.time()-t0:.2f} s")

    print(f"Querying for {n_in} segments ...")
    t0 = time.time()
    good = ~bad_loc
    seg_lon = seg.loc[good, "Longitude"].astype(float).values
    seg_lat = seg.loc[good, "Latitude"].astype(float).values
    seg_xyz = lonlat_to_xyz(seg_lon, seg_lat)

    chord, _ = tree.query(seg_xyz, k=1)
    # chord length on unit sphere -> great-circle distance in km
    dist_km = 2.0 * EARTH_R_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    print(f"  done in {time.time()-t0:.2f} s")

    seg["dist_to_Antarctica_km"] = np.nan
    seg.loc[good, "dist_to_Antarctica_km"] = dist_km

    out_path = META_WITH_ENV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_path, index=False)

    # ---------------- summary ----------------
    s = seg["dist_to_Antarctica_km"].describe()
    print("\n===== SUMMARY =====")
    print(f"  rows written : {len(seg)} -> {out_path}")
    print(f"  dist_to_Antarctica_km stats:")
    print(s)
    print()
    print(f"  segments within 100 km of Antarctica: "
          f"{int((seg['dist_to_Antarctica_km'] < 100).sum())}")
    print(f"  segments within 500 km of Antarctica: "
          f"{int((seg['dist_to_Antarctica_km'] < 500).sum())}")
    print(f"\nTotal wall time: {time.time() - t_total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
