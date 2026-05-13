import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path

    return Path, json, mo, pd


@app.cell
def header(mo):
    mo.md("""
    # Build `topic_features` and `topic_meta`
    "
        "Reads every `H_normalized.csv` + `meta.json` under `data/nmf_per_tow/`, "
        "assembles the two reusable tables and writes them to "
        "`data/processed/topic_features.csv` and `data/processed/topic_meta.csv`.
    """)
    return


@app.cell(hide_code=True)
def load_topics(Path, json, mo, pd):
    _root = Path("data/nmf_per_tow")

    _all_topics = []
    _meta_cols = [
        "group_id", "ship", "n_segments", "n_tows_merged",
        "best_k", "component", "period_start", "period_end", "folder",
    ]

    for _folder in sorted(_root.iterdir()):
        if not _folder.is_dir():
            continue
        _h_path = _folder / "H_normalized.csv"
        _meta_path = _folder / "meta.json"
        if not _h_path.exists() or not _meta_path.exists():
            continue

        _h = pd.read_csv(_h_path)
        with open(_meta_path) as _f:
            _gmeta = json.load(_f)

        _gid = _gmeta["group_id"]
        _comp_col = _h.columns[0]
        _species_cols = _h.columns[1:]

        for _, _row in _h.iterrows():
            _topic_row = _row[_species_cols].to_dict()
            _topic_row["group_id"] = _gid
            _topic_row["ship"] = _gmeta["ship"]
            _topic_row["n_segments"] = _gmeta["n_segments"]
            _topic_row["n_tows_merged"] = _gmeta["n_tows_merged"]
            _topic_row["best_k"] = _gmeta["best_k"]
            _topic_row["component"] = _row[_comp_col]
            _topic_row["period_start"] = _gmeta["period_start"][:10]
            _topic_row["period_end"] = _gmeta["period_end"][:10]
            # Tow year derived from period_start (fallback to period_end)
            try:
                _topic_row["year"] = int(_gmeta["period_start"][:4])
            except (ValueError, TypeError):
                try:
                    _topic_row["year"] = int(_gmeta["period_end"][:4])
                except (ValueError, TypeError):
                    _topic_row["year"] = None
            _topic_row["folder"] = _folder.name
            _all_topics.append(_topic_row)

    _topics_df = pd.DataFrame(_all_topics).fillna(0.0)

    _meta_cols_full = _meta_cols + ["year"]
    topic_meta = _topics_df[_meta_cols_full].copy().reset_index(drop=True)
    _species_cols_all = [c for c in _topics_df.columns if c not in _meta_cols_full]
    topic_features = (
        _topics_df[_species_cols_all].copy().astype(float).reset_index(drop=True)
    )

    # Drop species that are zero in every topic
    _nonzero = topic_features.sum(axis=0) > 0
    topic_features = topic_features.loc[:, _nonzero]

    # Derived columns on topic_meta
    topic_meta["topic_label"] = (
        "G" + topic_meta["group_id"].astype(str) + "_" + topic_meta["component"]
    )
    topic_meta["dominant_species"] = topic_features.idxmax(axis=1).values
    topic_meta["topic_idx"] = topic_meta.index

    mo.md(
        f"Built **{len(topic_meta)} topics** from "
        f"**{topic_meta['group_id'].nunique()} tow groups** across "
        f"**{topic_meta['year'].nunique()} years**, "
        f"**{topic_features.shape[1]} species** retained."
    )
    return topic_features, topic_meta


@app.cell
def preview_meta(mo, topic_meta):
    mo.vstack([mo.md("### `topic_meta` (first 20 rows)"), mo.ui.table(topic_meta.head(20))])
    return


@app.cell
def preview_features(mo, topic_features):
    mo.vstack([
        mo.md(
            f"### `topic_features` ({topic_features.shape[0]} rows × "
            f"{topic_features.shape[1]} species cols, first 20 rows × 12 cols)"
        ),
        mo.ui.table(topic_features.iloc[:20, :12]),
    ])
    return


@app.cell
def save_controls(mo):
    save_btn = mo.ui.button(label="Save topic_features + topic_meta to CSV", kind="success")
    save_btn
    return (save_btn,)


@app.cell
def save_tables(Path, mo, save_btn, topic_features, topic_meta):
    save_btn  # depend on the button so the cell re-runs on click

    _out_dir = Path("data/processed")
    _out_dir.mkdir(parents=True, exist_ok=True)

    _meta_path = _out_dir / "topic_meta.csv"
    _feat_path = _out_dir / "topic_features.csv"

    if save_btn.value:
        topic_meta.to_csv(_meta_path, index=False)
        topic_features.to_csv(_feat_path, index=False)
        _msg = mo.md(
            f"\u2705 Saved:\n"
            f"- `{_meta_path}` ({len(topic_meta)} rows, {topic_meta.shape[1]} cols)\n"
            f"- `{_feat_path}` ({topic_features.shape[0]} rows, {topic_features.shape[1]} species cols)"
        )
    else:
        _msg = mo.md(
            f"*Click the button above to write:*\n"
            f"- `{_meta_path}`\n"
            f"- `{_feat_path}`"
        )
    _msg
    return


if __name__ == "__main__":
    app.run()
