#!/usr/bin/env python3
"""
Build nmf_data.js and nmf_weights.js for the D3 explorer.

nmf_data.js   — top-5 species per topic, grouped by austral season (for the panel)
nmf_weights.js — per-segment NMF weights keyed by Segment_ID (for map colouring)
"""
import json
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NMF_DIR = BASE / "data" / "nmf_per_tow"
DATA = BASE / "data"
FIGURES = BASE / "figures"


def get_season(period_start: str) -> str:
    year = int(period_start[:4])
    month = int(period_start[5:7])
    if month >= 11:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def read_h_normalized(group_dir: Path):
    path = group_dir / "H_normalized.csv"
    topics = []
    with open(path) as f:
        reader = csv.DictReader(f)
        species_cols = [c for c in reader.fieldnames if c != "component"]
        for row in reader:
            comp_name = row["component"]
            weights = []
            for sp in species_cols:
                w = float(row[sp])
                if w > 0:
                    weights.append((sp, w))
            weights.sort(key=lambda x: -x[1])
            top5 = [[sp, round(w, 4)] for sp, w in weights[:5]]
            topics.append({"id": comp_name, "species": top5})
    return topics


def main():
    summary_path = NMF_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    by_season = {}

    for result in summary["all_results"]:
        gid = result["group_id"]
        ship = result["ship"]
        season = get_season(result["period_start"])
        k = result["best_k"]
        n_segs = result["n_segments"]

        group_dir = None
        for d in NMF_DIR.iterdir():
            if d.is_dir() and d.name.startswith(f"group_{gid:04d}_"):
                group_dir = d
                break

        if group_dir is None or not (group_dir / "H_normalized.csv").exists():
            print(f"  Warning: missing H_normalized for group {gid}, skipping")
            continue

        topics = read_h_normalized(group_dir)

        entry = {
            "g": gid,
            "ship": ship,
            "k": k,
            "n": n_segs,
            "topics": topics,
        }

        if season not in by_season:
            by_season[season] = []
        by_season[season].append(entry)

    out_path = FIGURES / "nmf_data.js"
    with open(out_path, "w") as f:
        f.write("window.NMF_DATA=")
        json.dump(by_season, f, separators=(",", ":"))
        f.write(";")

    n_topics = sum(
        sum(e["k"] for e in groups) for groups in by_season.values()
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"nmf_data.js: {size_kb:.0f} KB")
    print(f"  {len(by_season)} seasons, {sum(len(g) for g in by_season.values())} groups, {n_topics} topics")

    # ── Build per-segment NMF weights (nmf_weights.js) ──────────────────
    # Read segment metadata to get ordered Segment_IDs matching cpr_data.js
    import pandas as pd
    meta = pd.read_csv(DATA / "metadata" / "segment_metadata.csv")
    valid_mask = (
        meta["Latitude"].notna() & meta["Longitude"].notna()
        & meta["Segment_Length"].notna() & meta["Year"].notna()
    )
    meta = meta[valid_mask]
    seg_ids = meta["Segment_ID"].tolist()
    seg_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

    # For each segment: [group_id, w1, w2, ...] or null
    n_segs = len(seg_ids)
    weights = [None] * n_segs
    matched = 0

    for result in summary["all_results"]:
        gid = result["group_id"]

        group_dir = None
        for d in NMF_DIR.iterdir():
            if d.is_dir() and d.name.startswith(f"group_{gid:04d}_"):
                group_dir = d
                break
        if group_dir is None:
            continue

        w_path = group_dir / "W_normalized.csv"
        if not w_path.exists():
            continue

        with open(w_path) as f:
            reader = csv.DictReader(f)
            comp_cols = [c for c in reader.fieldnames if c != "Segment_ID"]
            for row in reader:
                sid = row["Segment_ID"]
                if sid in seg_to_idx:
                    idx = seg_to_idx[sid]
                    ws = [round(float(row[c]), 4) for c in comp_cols]
                    weights[idx] = [gid] + ws
                    matched += 1

    w_path = FIGURES / "nmf_weights.js"
    with open(w_path, "w") as f:
        f.write("window.NMF_WEIGHTS=")
        json.dump(weights, f, separators=(",", ":"))
        f.write(";")

    size_kb = w_path.stat().st_size / 1024
    print(f"nmf_weights.js: {size_kb:.0f} KB")
    print(f"  {matched}/{n_segs} segments matched ({matched/n_segs*100:.1f}%)")


if __name__ == "__main__":
    main()
