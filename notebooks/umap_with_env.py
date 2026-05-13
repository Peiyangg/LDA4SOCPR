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
    import plotly.express as px
    import plotly.graph_objects as go
    import umap

    return Path, go, mo, np, pd, px, umap


@app.cell
def header(mo):
    mo.md("""
    # UMAP of NMF topics, enriched with environmental context
    "
        "For every topic we identify its **representative segments** (those with "
        "topic-weight > 0.8 in `W_normalized.csv`), then aggregate the **5 numeric** "
        "per-segment environmental features (`SST`, `IC`, `dist_to_PF_weekly_km`, "
        "`dist_to_PF_clim_km`, `dist_to_Antarctica_km`) as **mean** and **sd**.
    "
        "PF-distance is split by source: `weekly` (2002-06-08 .. 2014-02-22) and "
        "`climatology` (everything else). The result is appended to `topic_meta`, "
        "so the UMAP scatter can be coloured by any of these new fields.
    """)
    return


@app.cell(hide_code=True)
def load_topics(Path, mo, pd):
    _feat = Path("data/processed/topic_features.csv")
    _meta = Path("data/processed/topic_meta.csv")

    if not _feat.exists() or not _meta.exists():
        topic_features = None
        topic_meta = None
        _msg = mo.md(
            f"\u26a0 Cannot find `{_feat}` or `{_meta}`.  "
            "Run `notebooks/build_topic_tables.py` first to materialise them."
        )
    else:
        topic_features = pd.read_csv(_feat)
        topic_meta = pd.read_csv(_meta)
        _msg = mo.md(
            f"Loaded **{len(topic_meta)} topics** \u00d7 "
            f"**{topic_features.shape[1]} species**."
        )
    _msg
    return topic_features, topic_meta


@app.cell(hide_code=True)
def load_segments_env(Path, mo, pd):
    _p = Path("data/metadata/segment_metadata_with_env.csv")
    if not _p.exists():
        segments_env = None
        _msg = mo.md(f"\u26a0 Cannot find `{_p}`. Build it with the assign scripts first.")
    else:
        segments_env = pd.read_csv(_p)
        _msg = mo.md(
            f"Loaded **{len(segments_env)} segments** with environmental cols: "
            f"`SST`, `IC`, `dist_to_PF_weekly_km`, `dist_to_PF_clim_km`, `dist_to_Antarctica_km`."
        )
    _msg
    return (segments_env,)


@app.cell(hide_code=True)
def threshold_control(mo):
    weight_threshold = mo.ui.slider(
        start=0.5, stop=0.95, value=0.8, step=0.05,
        label="Representative-segment weight threshold",
        show_value=True,
    )
    weight_threshold
    return (weight_threshold,)


@app.cell(hide_code=True)
def enrich_topic_meta(
    Path,
    mo,
    np,
    pd,
    segments_env,
    topic_meta,
    weight_threshold,
):
    if topic_meta is None or segments_env is None:
        topic_meta_env = None
        _msg = mo.md("*Waiting for `topic_meta` and `segments_env` to load.*")
    else:
        env_cols = [
            "SST", "IC",
            "dist_to_PF_weekly_km", "dist_to_PF_clim_km",
            "dist_to_Antarctica_km",
        ]
        thr = float(weight_threshold.value)

        _root = Path("data/nmf_per_tow")
        records = {}
        n_groups_seen = 0
        n_groups_missing_w = 0

        for _folder in sorted(_root.iterdir()):
            if not _folder.is_dir():
                continue
            _w_path = _folder / "W_normalized.csv"
            if not _w_path.exists():
                n_groups_missing_w += 1
                continue

            _w = pd.read_csv(_w_path)
            if "Segment_ID" not in _w.columns:
                continue
            n_groups_seen += 1

            _comp_cols = [c for c in _w.columns if c.startswith("component_")]
            _env_sub = segments_env.loc[
                segments_env["Segment_ID"].isin(_w["Segment_ID"]),
                ["Segment_ID"] + env_cols,
            ]
            _merged = _w[["Segment_ID"] + _comp_cols].merge(_env_sub, on="Segment_ID", how="left")

            _meta_path = _folder / "meta.json"
            if _meta_path.exists():
                import json as _json
                with open(_meta_path) as _f:
                    _gid = _json.load(_f)["group_id"]
            else:
                _gid = _folder.name

            for _comp in _comp_cols:
                _rep_mask = _merged[_comp] > thr
                _n_rep = int(_rep_mask.sum())
                _stats = {"group_id": _gid, "component": _comp, "n_representative": _n_rep}
                for _c in env_cols:
                    vals = _merged.loc[_rep_mask, _c]
                    _stats[f"{_c}_mean"] = float(np.nanmean(vals)) if _n_rep else np.nan
                    _stats[f"{_c}_sd"]   = float(np.nanstd(vals, ddof=1)) if _n_rep > 1 else np.nan
                records[(_gid, _comp)] = _stats

        _agg = pd.DataFrame(records.values())

        _meta = topic_meta.copy()
        _meta["group_id"] = _meta["group_id"].astype(_agg["group_id"].dtype)
        _meta["component"] = _meta["component"].astype(str)
        _agg["component"] = _agg["component"].astype(str)

        topic_meta_env = _meta.merge(_agg, on=["group_id", "component"], how="left")

        _new_cols = [c for c in topic_meta_env.columns if c not in topic_meta.columns]
        _n_topics_with_rep = int((topic_meta_env["n_representative"].fillna(0) > 0).sum())
        _msg = mo.md(
            f"Enriched `topic_meta` with **{len(_new_cols)} new cols** "
            f"(threshold W > {thr}).\n\n"
            f"- groups processed: {n_groups_seen}  (missing W: {n_groups_missing_w})\n"
            f"- topics with at least one representative segment: "
            f"{_n_topics_with_rep} / {len(topic_meta_env)}\n"
            f"- new columns: `{', '.join(_new_cols)}`"
        )
    _msg
    return (topic_meta_env,)


@app.cell(hide_code=True)
def preview_topic_meta_env(mo, topic_meta_env):
    if topic_meta_env is None:
        _v = mo.md("")
    else:
        _show_cols = (
            ["topic_label", "group_id", "component", "best_k", "ship",
             "dominant_species", "year", "n_representative"]
            + [c for c in topic_meta_env.columns if c.endswith("_mean") or c.endswith("_sd")]
        )
        _show_cols = [c for c in _show_cols if c in topic_meta_env.columns]
        _v = mo.vstack([
            mo.md("### Enriched `topic_meta` (first 25 rows)"),
            mo.ui.table(topic_meta_env[_show_cols].head(25)),
        ])
    # _v
    return


@app.cell(hide_code=True)
def umap_controls(mo):
    n_neighbors_slider = mo.ui.slider(start=5, stop=50, value=15, step=5, label="n_neighbors")
    min_dist_slider = mo.ui.slider(start=0.0, stop=1.0, value=0.1, step=0.05, label="min_dist")
    metric_dropdown = mo.ui.dropdown(
        options=["cosine", "euclidean", "correlation", "manhattan"],
        value="cosine", label="metric",
    )
    mo.md("### UMAP Parameters")
    mo.hstack([n_neighbors_slider, min_dist_slider, metric_dropdown])
    return metric_dropdown, min_dist_slider, n_neighbors_slider


@app.cell(hide_code=True)
def run_umap(
    metric_dropdown,
    min_dist_slider,
    mo,
    n_neighbors_slider,
    topic_features,
    topic_meta_env,
    umap,
):
    if topic_features is None or topic_meta_env is None:
        umap_df = None
        _msg = mo.md("*Waiting for topic_features / topic_meta_env.*")
    else:
        _reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors_slider.value,
            min_dist=min_dist_slider.value,
            metric=metric_dropdown.value,
            random_state=42,
        )
        _emb = _reducer.fit_transform(topic_features.values)

        umap_df = topic_meta_env.copy()
        umap_df["umap_1"] = _emb[:, 0]
        umap_df["umap_2"] = _emb[:, 1]
        _msg = mo.md(
            f"UMAP done: n_neighbors={n_neighbors_slider.value}, "
            f"min_dist={min_dist_slider.value}, metric={metric_dropdown.value}"
        )
    _msg
    return (umap_df,)


@app.cell(hide_code=True)
def color_by_control(mo, topic_features, topic_meta_env):
    if topic_meta_env is None or topic_features is None:
        color_by = mo.ui.dropdown(
            options=["dominant_species"], value="dominant_species", label="Color by"
        )
        species_dropdown = mo.ui.dropdown(options=["-"], value="-", label="Species")
    else:
        _new_env = [c for c in topic_meta_env.columns
                    if c.endswith("_mean") or c.endswith("_sd")]
        _base = [
            "dominant_species", "ship", "best_k", "component",
            "year", "n_segments", "n_tows_merged", "n_representative",
            "species_weight",
        ]
        _opts = [c for c in _base
                 if (c in topic_meta_env.columns) or (c == "species_weight")]
        _opts += _new_env
        color_by = mo.ui.dropdown(
            options=_opts, value="dominant_species", label="Color by",
            searchable=True,
        )

        # Top-10 species by total weight across all topics
        _species_sorted = (
            topic_features.sum(axis=0).sort_values(ascending=False).head(10)
            .index.tolist()
        )
        species_dropdown = mo.ui.dropdown(
            options=_species_sorted,
            value=_species_sorted[0] if _species_sorted else None,
            label="Species (top 10, used when Color by = species_weight)",
            searchable=True,
        )
    mo.md("### Color the UMAP scatter")
    mo.hstack([color_by, species_dropdown])
    return color_by, species_dropdown


@app.cell(hide_code=True)
def umap_scatter(color_by, mo, px, species_dropdown, topic_features, umap_df):
    if umap_df is None:
        _out = mo.md("*UMAP not ready.*")
    else:
        _df = umap_df.copy()
        _choice = color_by.value

        # ---- choose the column we'll actually colour by
        if (_choice == "species_weight" and species_dropdown.value is not None
                and species_dropdown.value in topic_features.columns):
            _sp = species_dropdown.value
            _df["_species_weight"] = topic_features[_sp].values
            _color_col = "_species_weight"
            _title = f"UMAP of NMF topics \u2014 colored by weight of \u201c{_sp}\u201d"
            _colorbar_title = f"{_sp}<br>weight"
        elif _choice == "dominant_species" and "dominant_species" in _df.columns:
            _top10 = _df["dominant_species"].value_counts().head(10).index.tolist()
            _df["dominant_species_top10"] = _df["dominant_species"].where(
                _df["dominant_species"].isin(_top10), other="Others"
            )
            _color_col = "dominant_species_top10"
            _title = (
                "UMAP of NMF topics \u2014 colored by dominant_species "
                "(top 10 + Others)"
            )
            _colorbar_title = None
        else:
            _color_col = _choice
            _title = f"UMAP of NMF topics \u2014 colored by {_color_col}"
            _colorbar_title = _color_col

        _is_num = _df[_color_col].dtype.kind in "fiu"
        _hover = [
            c for c in [
                "topic_label", "ship", "dominant_species", "best_k",
                "n_segments", "year", "n_representative",
                "SST_mean", "SST_sd",
                "IC_mean", "IC_sd",
                "dist_to_PF_weekly_km_mean", "dist_to_PF_weekly_km_sd",
                "dist_to_PF_clim_km_mean", "dist_to_PF_clim_km_sd",
                "dist_to_Antarctica_km_mean", "dist_to_Antarctica_km_sd",
            ]
            if c in _df.columns
        ]

        if _is_num:
            # Continuous colour. Decide sequential vs diverging.
            _ok_df = _df[_df[_color_col].notna()].copy()
            _nan_df = _df[_df[_color_col].isna()].copy()
            _vmin = float(_ok_df[_color_col].min()) if len(_ok_df) else 0.0
            _vmax = float(_ok_df[_color_col].max()) if len(_ok_df) else 0.0
            _is_div = (_vmin < 0.0) and (_vmax > 0.0)
            _scale = "RdYlBu" if _is_div else "Blues"

            _fig = px.scatter(
                _ok_df,
                x="umap_1", y="umap_2",
                color=_color_col,
                color_continuous_scale=_scale,
                hover_data=_hover,
                title=_title,
                width=1000, height=650,
            )
            if _is_div:
                _abs = max(abs(_vmin), abs(_vmax))
                _fig.update_coloraxes(cmin=-_abs, cmax=_abs, cmid=0.0)
            if _colorbar_title:
                _fig.update_layout(coloraxis_colorbar=dict(title=_colorbar_title))

            # Add NaN points as grey, drawn UNDER the coloured ones so they
            # are visible but don't fight for attention.
            if len(_nan_df):
                _fig.add_scatter(
                    x=_nan_df["umap_1"], y=_nan_df["umap_2"],
                    mode="markers",
                    name="missing",
                    marker=dict(
                        size=8, color="lightgrey", opacity=0.85,
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertext=_nan_df["topic_label"]
                        if "topic_label" in _nan_df.columns else None,
                    hoverinfo="text+name",
                    showlegend=True,
                )
                # move that trace to the bottom
                _fig.data = (_fig.data[-1],) + _fig.data[:-1]
        else:
            # Categorical colour. For dominant_species_top10 force "Others" -> grey.
            _cmap = {"Others": "lightgrey"} if _color_col == "dominant_species_top10" else None
            _fig = px.scatter(
                _df,
                x="umap_1", y="umap_2",
                color=_color_col,
                color_discrete_map=_cmap,
                hover_data=_hover,
                title=_title,
                width=2000, height=1000,
            )

        # Common styling — WHITE background, dark axes/text.
        _fig.update_traces(
            marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="white")),
            selector=dict(mode="markers"),
        )
        _fig.update_layout(
            template="plotly_white",
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="monospace", color="#222"),
            legend=dict(font=dict(size=10), itemsizing="constant"),
            dragmode="select",
            width=2000, height=1000,
        )
        _fig.update_xaxes(showgrid=True, gridcolor="#eaeaea", zeroline=False)
        _fig.update_yaxes(showgrid=True, gridcolor="#eaeaea", zeroline=False)
        umap_plot = mo.ui.plotly(_fig)
    umap_plot
    return (umap_plot,)


@app.cell(hide_code=True)
def selection_composition(
    go,
    mo,
    np,
    topic_features,
    topic_meta_env,
    umap_plot,
):
    """Show top-N species barplots for topics selected (box/lasso) on the UMAP."""
    from plotly.subplots import make_subplots
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list

    TOP_N = 5

    if umap_plot is None or topic_meta_env is None or topic_features is None:
        _out = mo.md("*UMAP not ready.*")
    else:
        _selected = umap_plot.value or []
        _labels = [pt.get("topic_label") for pt in _selected
                   if isinstance(pt, dict) and pt.get("topic_label")]

        if not _labels:
            _out = mo.md(
                "*Box-select or lasso-select topics in the UMAP scatter above "
                "to see their top species barplots here.*"
            )
        else:
            _label_to_row = {lab: i for i, lab in enumerate(topic_meta_env["topic_label"])}
            _selected_idx = [_label_to_row[l] for l in _labels if l in _label_to_row]
            if not _selected_idx:
                _out = mo.md("*Selected topics not found in topic_meta_env.*")
            else:
                # For each selected topic, take its top-N species by weight.
                _topics = []
                for _i in _selected_idx:
                    _row = topic_features.iloc[_i]
                    _top = _row.nlargest(TOP_N)
                    _topics.append({
                        "label": topic_meta_env.iloc[_i]["topic_label"],
                        "species": dict(_top),
                    })

                # Order topics by cosine-similarity of their species vector
                _all_sp = sorted({sp for t in _topics for sp in t["species"]})
                _sp_idx = {sp: i for i, sp in enumerate(_all_sp)}
                _vecs = np.zeros((len(_topics), len(_all_sp)))
                for _i, _t in enumerate(_topics):
                    for _sp, _w in _t["species"].items():
                        _vecs[_i, _sp_idx[_sp]] = _w
                if len(_topics) > 2:
                    _d = np.nan_to_num(pdist(_vecs, metric="cosine"), nan=1.0)
                    _order = leaves_list(linkage(_d, method="average")).tolist()
                else:
                    _order = list(range(len(_topics)))

                _n = len(_topics)
                _ncols = min(6, _n)
                _nrows = -(-_n // _ncols)

                _fig = make_subplots(
                    rows=_nrows, cols=_ncols,
                    subplot_titles=[_topics[i]["label"] for i in _order],
                    vertical_spacing=max(0.06, 0.18 / _nrows),
                    horizontal_spacing=0.08,
                )

                # Consistent per-species colours, shared across subplots
                _palette = [
                    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                ]
                _sp_colors = {sp: _palette[i % len(_palette)] for i, sp in enumerate(_all_sp)}
                _shown_sp = set()

                for _pos, _idx in enumerate(_order):
                    _t = _topics[_idx]
                    _row = _pos // _ncols + 1
                    _col = _pos % _ncols + 1
                    for _si, (_sp, _w) in enumerate(_t["species"].items()):
                        _show = _sp not in _shown_sp
                        _fig.add_trace(
                            go.Bar(
                                x=[_si], y=[_w],
                                marker_color=_sp_colors[_sp],
                                name=_sp, legendgroup=_sp, showlegend=_show,
                                text=[f"{_w:.2f}"], textposition="outside",
                                width=0.6,
                            ),
                            row=_row, col=_col,
                        )
                        if _show:
                            _shown_sp.add(_sp)
                    _fig.update_xaxes(showticklabels=False, row=_row, col=_col)
                    _fig.update_yaxes(range=[0, 1], gridcolor="#eaeaea", row=_row, col=_col)

                _fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white", plot_bgcolor="white",
                    height=240 * _nrows + 90, width=320 * _ncols + 220,
                    font=dict(family="monospace", size=10, color="#222"),
                    title=f"Top-{TOP_N} species per selected topic (ordered by similarity)",
                    margin=dict(t=80, b=20, r=210),
                    legend=dict(
                        title="Species", font=dict(size=10),
                        orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    ),
                    barmode="group",
                )
                _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def save_controls(mo):
    save_btn = mo.ui.button(label="Save enriched topic_meta to CSV", kind="success")
    save_btn
    return (save_btn,)


@app.cell
def save_enriched(Path, mo, save_btn, topic_meta_env):
    save_btn  # depend on the button

    if topic_meta_env is None:
        _msg = mo.md("*Nothing to save yet.*")
    else:
        _out = Path("data/processed/topic_meta_env.csv")
        if save_btn.value:
            _out.parent.mkdir(parents=True, exist_ok=True)
            topic_meta_env.to_csv(_out, index=False)
            _msg = mo.md(
                f"\u2705 Saved enriched topic_meta to `{_out}`  "
                f"({len(topic_meta_env)} rows, {topic_meta_env.shape[1]} cols)"
            )
        else:
            _msg = mo.md(f"*Click the button to save to `{_out}`.*")
    _msg
    return


if __name__ == "__main__":
    app.run()
