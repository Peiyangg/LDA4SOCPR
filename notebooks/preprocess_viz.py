#!/usr/bin/env python3
"""Preprocess CPR data for D3 visualization."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures" / "cpr_data.json"

print("Loading data...")
meta = pd.read_csv(BASE / "metadata" / "segment_metadata.csv")
abund = pd.read_csv(BASE / "abundance_processed.csv", index_col="Segment_ID")
groups_df = pd.read_csv(BASE / "abundance_pinkerton2020.csv", index_col="Segment_ID")

print(f"Meta: {len(meta)} rows | Abund: {len(abund)} rows | Groups: {len(groups_df)} rows")

# --- Shannon diversity (vectorised) ---
print("Calculating Shannon diversity...")
arr = abund.values.astype(float)
totals = arr.sum(axis=1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    p = np.where(totals > 0, arr / totals, 0.0)
    logp = np.where(p > 0, np.log(p), 0.0)
shannon_vals = -(p * logp).sum(axis=1)
shannon = pd.Series(shannon_vals, index=abund.index, name="shannon")

# --- Month ordering (no June in data) ---
months = [
    "January", "February", "March", "April", "May",
    "July", "August", "September", "October", "November", "December",
]
month_map = {m: i for i, m in enumerate(months)}

# --- Group columns ---
group_cols = [c for c in groups_df.columns]

# --- Build records ---
print("Building output records...")
records = []
for _, row in meta.iterrows():
    sid = row["Segment_ID"]
    try:
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        seg_len = float(row["Segment_Length"])
        year = int(row["Year"])
    except (ValueError, TypeError):
        continue

    sh = float(shannon.get(sid, 0.0))
    if np.isnan(sh):
        sh = 0.0

    m_idx = month_map.get(str(row["Month"]).strip(), -1)

    if sid in groups_df.index:
        g_row = groups_df.loc[sid]
        g = [int(g_row[c]) if not pd.isna(g_row[c]) else 0 for c in group_cols]
    else:
        g = [0] * len(group_cols)

    try:
        tow_no = int(float(row["Tow_Number"])) if pd.notna(row["Tow_Number"]) else 0
    except (ValueError, TypeError):
        tow_no = 0

    # Record layout (compact array):
    # [lat, lon, year, month_idx, seg_len, shannon, ship, tow_no, seg_no, date, time, [groups...]]
    records.append([
        round(lat, 4),
        round(lon, 4),
        year,
        m_idx,
        round(seg_len, 3),
        round(sh, 4),
        str(row["Ship_Code"]).strip(),
        tow_no,
        int(row["Segment_No."]),
        str(row["Date"]).strip(),
        str(row["Time"]).strip(),
        g,
    ])

output = {
    "months": months,
    "groups": group_cols,
    "segments": records,
}

JS_CPR   = Path(__file__).parent / "figures" / "cpr_data.js"
JS_WORLD = Path(__file__).parent / "figures" / "world_data.js"

# --- Write CPR data as a JS variable (works with file:// protocol) ---
print(f"Writing {len(records):,} segments to {JS_CPR} ...")
with open(JS_CPR, "w") as f:
    f.write("window.CPR_DATA=")
    json.dump(output, f, separators=(",", ":"))
    f.write(";")

import os
size_mb = os.path.getsize(JS_CPR) / 1024 / 1024
print(f"cpr_data.js: {size_mb:.1f} MB")

# --- Download world-atlas land polygons (110m) for basemap ---
print("Downloading world-atlas land-110m.json ...")
import urllib.request
url = "https://cdn.jsdelivr.net/npm/world-atlas@2/land-110m.json"
try:
    with urllib.request.urlopen(url, timeout=15) as resp:
        world_data = json.loads(resp.read())
    with open(JS_WORLD, "w") as f:
        f.write("window.WORLD_DATA=")
        json.dump(world_data, f, separators=(",", ":"))
        f.write(";")
    print(f"world_data.js: {os.path.getsize(JS_WORLD)/1024:.0f} KB")
except Exception as e:
    print(f"Warning: could not download world-atlas ({e}). Map will show without land outlines.")
    with open(JS_WORLD, "w") as f:
        f.write("window.WORLD_DATA=null;")

print("Done!")

# --- LDA topic distributions (one JS file per K) ---
LDA_BASE = Path(__file__).parent / "LDA_gensim" / "lda_results" / "topic_distributions"
LDA_Ks   = [3, 7, 10, 16]

# Build a lookup: Segment_ID → index in records (same order as cpr_data.js)
seg_index = {r[11][0] if False else None: i for i, r in enumerate(records)}  # placeholder
# Actually: build from the records list (field 0 is not segment_id, we need to re-derive)
# Re-read metadata to get Segment_ID order matching records
meta2 = pd.read_csv(BASE / "metadata" / "segment_metadata.csv")
seg_to_idx = {}
idx = 0
for _, row in meta2.iterrows():
    sid = row["Segment_ID"]
    try:
        float(row["Latitude"]); float(row["Longitude"]); float(row["Segment_Length"]); int(row["Year"])
    except (ValueError, TypeError):
        continue
    seg_to_idx[sid] = idx
    idx += 1

n_segs = len(records)

for k in LDA_Ks:
    csv_path = LDA_BASE / f"K{k}_segment_topic_probs.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping")
        continue

    df_lda = pd.read_csv(csv_path, index_col="Segment_ID")
    topic_cols = list(df_lda.columns)  # K{k}_MC0 ... K{k}_MC{k-1}

    # Build array: probs[i] = [p0, p1, ...] for segment at index i
    probs = [[None] * k for _ in range(n_segs)]
    missing = 0
    for sid, i in seg_to_idx.items():
        if sid in df_lda.index:
            probs[i] = [round(float(df_lda.at[sid, c]), 4) for c in topic_cols]
        else:
            probs[i] = [round(1/k, 4)] * k  # fallback: uniform
            missing += 1

    js_out = Path(__file__).parent / "figures" / f"lda_k{k}.js"
    with open(js_out, "w") as f:
        f.write(f"window.LDA_K{k}=")
        json.dump(probs, f, separators=(",", ":"))
        f.write(";")

    sz = os.path.getsize(js_out) / 1024
    print(f"lda_k{k}.js: {sz:.0f} KB  (missing: {missing})")
