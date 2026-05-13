#!/usr/bin/env python3
"""
Preprocess CPR data into H3 hexagonal bins (res 2 + res 3),
one record per hex × year. Also convert Orsi fronts to GeoJSON.

Outputs:
  figures/hex_data.js        — D3 visualization data (year-specific hex records)
  figures/fronts_data.js     — Orsi fronts GeoJSON
  data/hex_features_res{2,3}.csv   — abundance per hex-year (272 species)
  data/hex_metadata_res{2,3}.csv   — metadata per hex-year (env, effort, etc.)
  data/hex_effort_res{2,3}.csv     — pivot: hex × year → sample count
"""
import pandas as pd
import numpy as np
import json
import h3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGURES = BASE / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# 1. SOUTHERN OCEAN FRONTS → GeoJSON
# ─────────────────────────────────────────────────────────────────────────────
FRONT_NAMES = {
    "stf": "Subtropical Front",
    "saf": "Subantarctic Front",
    "pf": "Polar Front",
    "saccf": "Southern ACC Front",
    "sbdy": "Southern Boundary",
}

FRONT_COLORS = {
    "stf": "#e74c3c",
    "saf": "#e67e22",
    "pf": "#2ecc71",
    "saccf": "#3498db",
    "sbdy": "#9b59b6",
}


def parse_front_file(path):
    segments = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                if current:
                    segments.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                current.append([float(parts[0]), float(parts[1])])
    if current:
        segments.append(current)
    return segments


def build_fronts_geojson():
    features = []
    fronts_dir = DATA / "fronts"
    for key, name in FRONT_NAMES.items():
        path = fronts_dir / f"{key}.txt"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping {name}")
            continue
        segments = parse_front_file(path)
        features.append({
            "type": "Feature",
            "properties": {"id": key, "name": name, "color": FRONT_COLORS[key]},
            "geometry": {"type": "MultiLineString", "coordinates": segments},
        })
        print(f"  {name}: {len(segments)} segments, {sum(len(s) for s in segments)} points")
    return {"type": "FeatureCollection", "features": features}


# ─────────────────────────────────────────────────────────────────────────────
# 2. H3 HEXAGONAL AGGREGATION — per hex × year
# ─────────────────────────────────────────────────────────────────────────────

def compute_shannon(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def safe_mean(series):
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else None


def build_hex_year_data(meta, groups_df, abund, resolution):
    """
    Aggregate CPR segments into H3 cells per austral season.
    Returns: hex_geo (dict of cell boundaries), hex_year_records (list of per-season dicts),
             feature_rows, metadata_rows, effort_rows.
    """
    print(f"\n  Building H3 res {resolution} hexagons (per austral season)...")

    meta = meta.copy()
    meta["h3_cell"] = [
        h3.latlng_to_cell(row["Latitude"], row["Longitude"], resolution)
        for _, row in meta.iterrows()
    ]

    group_cols = list(groups_df.columns)
    species_cols = list(abund.columns)

    hex_geo = {}
    hex_year_records = []
    feature_rows = []
    metadata_rows = []
    effort_rows = []

    for (cell, season), cell_meta in meta.groupby(["h3_cell", "Season"]):
        sids = cell_meta["Segment_ID"].values
        n_segments = len(sids)

        if cell not in hex_geo:
            boundary = h3.cell_to_boundary(cell)
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            mean_lat = float(cell_meta["Latitude"].mean())
            mean_lon = float(cell_meta["Longitude"].mean())
            hex_geo[cell] = {
                "boundary": coords,
                "center": [mean_lon, mean_lat],
            }

        # --- Pinkerton groups (for D3 viz) ---
        mask_g = groups_df.index.isin(sids)
        group_totals = groups_df.loc[mask_g].sum().values.tolist()

        # --- Raw species abundance (for features CSV) ---
        mask_a = abund.index.isin(sids)
        cell_abund = abund.loc[mask_a]
        species_totals = cell_abund.sum(axis=0)
        shannon = compute_shannon(species_totals.values)
        richness = int((species_totals.values > 0).sum())

        # --- Environmental metadata ---
        mean_temp = safe_mean(cell_meta["Water_Temperature"])
        mean_sal = safe_mean(cell_meta["Salinity"])
        mean_fluor = safe_mean(cell_meta["Fluorescence"])
        mean_pci = safe_mean(cell_meta["Phytoplankton_Colour_Index"])
        mean_par = safe_mean(cell_meta["Photosynthetically_Active_Radiation"])
        mean_sst = safe_mean(cell_meta["hadisst_sst"]) if "hadisst_sst" in cell_meta.columns else None
        mean_ice = safe_mean(cell_meta["hadisst_ice"]) if "hadisst_ice" in cell_meta.columns else None

        # --- D3 record ---
        hex_year_records.append({
            "cell": cell,
            "season": str(season),
            "n": n_segments,
            "sh": round(shannon, 4),
            "ri": richness,
            "g": [int(v) for v in group_totals],
            "t": round(mean_temp, 2) if mean_temp is not None else None,
            "sa": round(mean_sal, 2) if mean_sal is not None else None,
            "fl": round(mean_fluor, 2) if mean_fluor is not None else None,
            "pc": round(mean_pci, 2) if mean_pci is not None else None,
            "pa": round(mean_par, 2) if mean_par is not None else None,
            "ss": round(mean_sst, 2) if mean_sst is not None else None,
            "ic": round(mean_ice, 3) if mean_ice is not None else None,
        })

        # --- Features CSV row (raw 272 species) ---
        feat_row = {"h3_cell": cell, "season": str(season)}
        for col in species_cols:
            feat_row[col] = int(species_totals[col])
        feature_rows.append(feat_row)

        # --- Metadata CSV row ---
        n_tows = int(cell_meta["Tow_Number"].nunique())
        n_ships = int(cell_meta["Ship_Code"].nunique())
        metadata_rows.append({
            "h3_cell": cell,
            "season": str(season),
            "center_lat": round(float(cell_meta["Latitude"].mean()), 4),
            "center_lon": round(float(cell_meta["Longitude"].mean()), 4),
            "n_segments": n_segments,
            "n_tows": n_tows,
            "n_ships": n_ships,
            "ships": ";".join(sorted(cell_meta["Ship_Code"].unique().tolist())),
            "season": ";".join(sorted(cell_meta["Season"].unique().tolist())),
            "months": ";".join(sorted(cell_meta["Month"].unique().tolist())),
            "mean_segment_length": round(float(cell_meta["Segment_Length"].mean()), 3),
            "total_abundance_sum": int(cell_meta["Total abundance"].sum()),
            "mean_total_abundance": round(float(cell_meta["Total abundance"].mean()), 2),
            "mean_total_plankton_corrected": round(float(cell_meta["Total_Plankton_Corrected"].mean()), 4),
            "mean_water_temperature": round(mean_temp, 2) if mean_temp is not None else None,
            "mean_salinity": round(mean_sal, 2) if mean_sal is not None else None,
            "mean_fluorescence": round(mean_fluor, 2) if mean_fluor is not None else None,
            "mean_phytoplankton_colour_index": round(mean_pci, 2) if mean_pci is not None else None,
            "mean_par": round(mean_par, 2) if mean_par is not None else None,
            "mean_hadisst_sst": round(mean_sst, 2) if mean_sst is not None else None,
            "mean_hadisst_ice": round(mean_ice, 3) if mean_ice is not None else None,
        })

        effort_rows.append({"h3_cell": cell, "season": str(season), "n_segments": n_segments})

    n_cells = len(hex_geo)
    n_records = len(hex_year_records)
    print(f"  → {n_cells} unique hexagons, {n_records} hex-year records at res {resolution}")
    return hex_geo, hex_year_records, feature_rows, metadata_rows, effort_rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # --- Fronts ---
    print("Building Southern Ocean fronts GeoJSON...")
    fronts_geojson = build_fronts_geojson()
    fronts_js = FIGURES / "fronts_data.js"
    with open(fronts_js, "w") as f:
        f.write("window.FRONTS_DATA=")
        json.dump(fronts_geojson, f, separators=(",", ":"))
        f.write(";")
    print(f"  → {fronts_js.name}: {fronts_js.stat().st_size / 1024:.0f} KB")

    # --- Load CPR data ---
    print("\nLoading CPR data...")
    meta = pd.read_csv(DATA / "metadata" / "segment_metadata.csv")
    abund = pd.read_csv(DATA / "abundance_processed.csv", index_col="Segment_ID")
    groups_df = pd.read_csv(DATA / "abundance_pinkerton2020.csv", index_col="Segment_ID")

    for col in ["Water_Temperature", "Salinity", "Fluorescence",
                 "Phytoplankton_Colour_Index", "Photosynthetically_Active_Radiation"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

    valid_mask = (
        meta["Latitude"].notna() & meta["Longitude"].notna()
        & meta["Segment_Length"].notna() & meta["Year"].notna()
    )
    meta = meta[valid_mask].copy()
    meta["Year"] = meta["Year"].astype(int)
    print(f"  {len(meta)} valid segments")

    env_path = DATA / "metadata" / "segment_env_data.csv"
    if env_path.exists():
        env = pd.read_csv(env_path)
        meta = meta.merge(env, on="Segment_ID", how="left")
        print(f"  Merged HadISST env data: SST={meta['hadisst_sst'].notna().sum()}, Ice={meta['hadisst_ice'].notna().sum()}")
    else:
        print("  Warning: segment_env_data.csv not found — run download_env_data.py first")

    group_cols = list(groups_df.columns)

    # --- Build hex-year data for both resolutions ---
    for resolution in [2, 3]:
        hex_geo, hex_year_records, feat_rows, meta_rows, effort_rows = \
            build_hex_year_data(meta, groups_df, abund, resolution)

        # --- D3 JS ---
        hex_output = {
            "groups": group_cols,
            "species": list(abund.columns),
            "geo": {cell: geo for cell, geo in hex_geo.items()},
            "data": hex_year_records,
        }
        hex_js = FIGURES / f"hex_res{resolution}.js"
        with open(hex_js, "w") as f:
            f.write(f"window.HEX_RES{resolution}=")
            json.dump(hex_output, f, separators=(",", ":"))
            f.write(";")
        size_mb = hex_js.stat().st_size / 1024 / 1024
        print(f"  → {hex_js.name}: {size_mb:.1f} MB")

        # --- Features CSV (raw 272 species per hex-year) ---
        df_feat = pd.DataFrame(feat_rows)
        feat_path = DATA / f"hex_features_res{resolution}.csv"
        df_feat.to_csv(feat_path, index=False)
        print(f"  → {feat_path.name}: {len(df_feat)} rows × {len(df_feat.columns)} cols")

        # --- Metadata CSV ---
        df_meta = pd.DataFrame(meta_rows)
        meta_path = DATA / f"hex_metadata_res{resolution}.csv"
        df_meta.to_csv(meta_path, index=False)
        print(f"  → {meta_path.name}: {len(df_meta)} rows × {len(df_meta.columns)} cols")

        # --- Effort pivot: hex × season ---
        df_effort = pd.DataFrame(effort_rows)
        pivot = df_effort.pivot_table(
            index="h3_cell", columns="season", values="n_segments", fill_value=0
        )
        pivot.columns = [str(c) for c in pivot.columns]
        effort_path = DATA / f"hex_effort_res{resolution}.csv"
        pivot.to_csv(effort_path)
        n_years_sampled = (pivot > 0).sum(axis=1)
        print(f"  → {effort_path.name}: {len(pivot)} hexes × {len(pivot.columns)} years")
        print(f"    Hexes sampled ≥10 years: {(n_years_sampled >= 10).sum()}")
        print(f"    Hexes sampled ≥20 years: {(n_years_sampled >= 20).sum()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
