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
    from scipy.spatial.distance import pdist, squareform

    return Path, go, json, mo, np, pd, pdist, px, squareform


@app.cell
def header(mo):
    mo.md("""
    # Jaccard Distance between NMF Topics
    "
        "Loads `topic_features` and `topic_meta` from `data/nmf_per_tow/`, then computes "
        "**binary Jaccard** (presence/absence) and **weighted Jaccard / Ruzicka** distances "
        "and shows their distributions.
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
            _topic_row["folder"] = _folder.name
            _all_topics.append(_topic_row)

    _topics_df = pd.DataFrame(_all_topics).fillna(0.0)
    topic_meta = _topics_df[_meta_cols].copy().reset_index(drop=True)
    _species_cols_all = [c for c in _topics_df.columns if c not in _meta_cols]
    topic_features = _topics_df[_species_cols_all].copy().astype(float).reset_index(drop=True)

    _nonzero = topic_features.sum(axis=0) > 0
    topic_features = topic_features.loc[:, _nonzero]

    topic_meta["topic_label"] = (
        "G" + topic_meta["group_id"].astype(str) + "_" + topic_meta["component"]
    )
    topic_meta["dominant_species"] = topic_features.idxmax(axis=1).values
    topic_meta["topic_idx"] = topic_meta.index

    mo.md(
        f"Loaded **{len(topic_meta)} topics** from "
        f"**{topic_meta['group_id'].nunique()} tow groups**, "
        f"**{topic_features.shape[1]} species**."
    )
    return topic_features, topic_meta


@app.cell
def controls(mo):
    presence_threshold = mo.ui.slider(
        start=0.0, stop=0.2, value=0.01, step=0.005,
        label="presence threshold (weight to count species as 'present')",
        show_value=True,
    )
    nbins = mo.ui.slider(start=20, stop=200, value=60, step=10, label="histogram bins")
    mo.md("### Parameters")
    mo.hstack([presence_threshold, nbins])
    return nbins, presence_threshold


@app.cell
def compute_jaccard(
    mo,
    np,
    pdist,
    presence_threshold,
    squareform,
    topic_features,
):
    _X = topic_features.values.astype(float)
    _thr = presence_threshold.value

    # Binary Jaccard: presence/absence after thresholding
    _X_bin = (_X > _thr).astype(bool)
    _jac_bin = pdist(_X_bin, metric="jaccard")

    # Weighted Jaccard (Ruzicka): 1 - sum(min) / sum(max)
    _n = _X.shape[0]
    _row_sums_max = _X.sum(axis=1)  # since min(x, x) = x
    _jac_w = np.empty(_n * (_n - 1) // 2, dtype=np.float64)
    _k = 0
    for _i in range(_n - 1):
        _xi = _X[_i]
        _xj = _X[_i + 1:]
        _mins = np.minimum(_xi, _xj).sum(axis=1)
        _maxs = np.maximum(_xi, _xj).sum(axis=1)
        _ratio = np.where(_maxs > 0, _mins / _maxs, 1.0)
        _block = 1.0 - _ratio
        _jac_w[_k:_k + _block.size] = _block
        _k += _block.size

    jac_bin = _jac_bin
    jac_weighted = _jac_w
    jac_bin_matrix = squareform(_jac_bin)
    jac_weighted_matrix = squareform(_jac_w)

    _n_zero_topics = int((_X_bin.sum(axis=1) == 0).sum())
    mo.md(
        f"Computed Jaccard distances over **{_n}** topics, "
        f"**{len(_jac_bin)}** pairs.\n"
        f"- presence threshold = {_thr}\n"
        f"- topics with **no species above threshold**: {_n_zero_topics}\n"
        f"- binary Jaccard range: [{_jac_bin.min():.3f}, {_jac_bin.max():.3f}], "
        f"mean = {_jac_bin.mean():.3f}\n"
        f"- weighted Jaccard range: [{jac_weighted.min():.3f}, {jac_weighted.max():.3f}], "
        f"mean = {jac_weighted.mean():.3f}"
    )
    return jac_bin, jac_bin_matrix, jac_weighted, jac_weighted_matrix


@app.cell
def distance_distribution(go, jac_bin, jac_weighted, mo, nbins, np):
    _bins = int(nbins.value)
    _edges = np.linspace(0, 1, _bins + 1)

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=jac_bin, xbins=dict(start=0, end=1, size=1 / _bins),
        name=f"binary Jaccard (mean={jac_bin.mean():.3f})",
        marker_color="#4e79a7", opacity=0.7,
    ))
    _fig.add_trace(go.Histogram(
        x=jac_weighted, xbins=dict(start=0, end=1, size=1 / _bins),
        name=f"weighted Jaccard / Ruzicka (mean={jac_weighted.mean():.3f})",
        marker_color="#f28e2b", opacity=0.7,
    ))

    for _name, _arr, _color in [
        ("binary mean", jac_bin, "#4e79a7"),
        ("weighted mean", jac_weighted, "#f28e2b"),
    ]:
        _fig.add_vline(
            x=float(_arr.mean()),
            line=dict(color=_color, dash="dash", width=1.5),
            annotation_text=_name,
            annotation_position="top",
        )

    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        title="Pairwise Jaccard distance distribution between NMF topics",
        xaxis_title="Jaccard distance (0 = identical, 1 = disjoint)",
        yaxis_title="number of topic pairs",
        barmode="overlay",
        width=1000, height=520,
        font=dict(family="monospace"),
        legend=dict(font=dict(size=10)),
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def per_topic_mean_distance(
    go,
    jac_bin_matrix,
    jac_weighted_matrix,
    mo,
    topic_meta,
):
    _n = jac_bin_matrix.shape[0]
    # Off-diagonal mean per topic (exclude self-distance)
    _mean_bin = (jac_bin_matrix.sum(axis=1)) / (_n - 1)
    _mean_w = (jac_weighted_matrix.sum(axis=1)) / (_n - 1)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_mean_bin, y=_mean_w, mode="markers",
        marker=dict(
            size=6, color=topic_meta["best_k"].astype(int),
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="best_k", thickness=12, len=0.6),
            line=dict(width=0.3, color="white"),
        ),
        text=topic_meta["topic_label"] + "<br>" + topic_meta["dominant_species"],
        hovertemplate="%{text}<br>mean binary=%{x:.3f}<br>mean weighted=%{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    _fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(200,200,200,0.4)", dash="dot", width=1),
    )
    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        title="Per-topic mean Jaccard distance (binary vs. weighted)",
        xaxis=dict(title="mean binary Jaccard", range=[0, 1]),
        yaxis=dict(title="mean weighted Jaccard", range=[0, 1], scaleanchor="x"),
        width=720, height=620,
        font=dict(family="monospace"),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def distance_heatmap(jac_weighted_matrix, mo, np, px, topic_meta):
    # Optional: a heatmap ordered by best_k for a quick block view
    _order = np.argsort(topic_meta["best_k"].values, kind="stable")
    _M = jac_weighted_matrix[np.ix_(_order, _order)]
    _labels = topic_meta["topic_label"].values[_order]

    _fig = px.imshow(
        _M,
        x=_labels, y=_labels,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Weighted Jaccard distance matrix (rows/cols ordered by best_k)",
        width=900, height=900,
    )
    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font=dict(family="monospace", size=8),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
