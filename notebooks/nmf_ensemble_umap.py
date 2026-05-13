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

    return Path, go, json, mo, np, pd, px, umap


@app.cell(hide_code=True)
def load_all_data(Path, json, mo, pd):
    mo.md("## Loading NMF topics + segment metadata")

    _root = Path("data/nmf_per_tow")
    _meta_full = pd.read_csv("data/metadata/segment_metadata.csv")

    _all_topics = []
    _all_segments = []

    for _folder in sorted(_root.iterdir()):
        if not _folder.is_dir():
            continue
        _h_path = _folder / "H_normalized.csv"
        _w_path = _folder / "W_normalized.csv"
        _meta_path = _folder / "meta.json"
        if not _h_path.exists() or not _meta_path.exists() or not _w_path.exists():
            continue

        _h = pd.read_csv(_h_path)
        _w = pd.read_csv(_w_path)
        with open(_meta_path) as _f:
            _gmeta = json.load(_f)

        _gid = _gmeta["group_id"]
        _ship = _gmeta["ship"]
        _comp_col = _h.columns[0]
        _species_cols = _h.columns[1:]

        for _, _row in _h.iterrows():
            _topic_row = _row[_species_cols].to_dict()
            _topic_row["group_id"] = _gid
            _topic_row["ship"] = _ship
            _topic_row["n_segments"] = _gmeta["n_segments"]
            _topic_row["n_tows_merged"] = _gmeta["n_tows_merged"]
            _topic_row["best_k"] = _gmeta["best_k"]
            _topic_row["component"] = _row[_comp_col]
            _topic_row["period_start"] = _gmeta["period_start"][:10]
            _topic_row["period_end"] = _gmeta["period_end"][:10]
            # Tow year: use period_start's year; fall back to period_end if invalid
            try:
                _topic_row["year"] = int(_gmeta["period_start"][:4])
            except (ValueError, TypeError):
                try:
                    _topic_row["year"] = int(_gmeta["period_end"][:4])
                except (ValueError, TypeError):
                    _topic_row["year"] = None
            _topic_row["folder"] = _folder.name
            _all_topics.append(_topic_row)

        _comp_cols = [c for c in _w.columns if c.startswith("component_")]
        _w_df = _w.copy()
        _w_df["group_id"] = _gid
        _w_df["dominant_component"] = _w_df[_comp_cols].idxmax(axis=1)
        _w_df["dominant_weight"] = _w_df[_comp_cols].max(axis=1)
        _all_segments.append(_w_df[["Segment_ID", "group_id", "dominant_component", "dominant_weight"] + _comp_cols])

    _topics_df = pd.DataFrame(_all_topics).fillna(0.0)
    segments_df = pd.concat(_all_segments, ignore_index=True)

    segments_df = segments_df.merge(
        _meta_full[["Segment_ID", "Latitude", "Longitude", "Ship_Code", "Date", "Month", "Year"]],
        on="Segment_ID", how="left",
    )

    _meta_cols = [
        "group_id", "ship", "n_segments", "n_tows_merged",
        "best_k", "component", "period_start", "period_end", "year", "folder",
    ]
    _species_cols_all = [c for c in _topics_df.columns if c not in _meta_cols]

    topic_meta = _topics_df[_meta_cols].copy()
    topic_features = _topics_df[_species_cols_all].copy().astype(float)

    _nonzero_mask = topic_features.sum(axis=0) > 0
    topic_features = topic_features.loc[:, _nonzero_mask]

    topic_meta = topic_meta.reset_index(drop=True)
    topic_features = topic_features.reset_index(drop=True)

    topic_meta["topic_label"] = (
        "G" + topic_meta["group_id"].astype(str) + "_" + topic_meta["component"]
    )
    topic_meta["dominant_species"] = topic_features.idxmax(axis=1).values
    topic_meta["topic_idx"] = topic_meta.index

    mo.md(
        f"Loaded **{len(topic_meta)} topics** from "
        f"**{topic_meta['group_id'].nunique()} tow groups**, "
        f"**{topic_features.shape[1]} species**, "
        f"**{len(segments_df)} segments** with lat/lon."
    )
    return topic_features, topic_meta


@app.cell
def summary_table(mo, topic_meta):
    _summary = topic_meta.groupby("best_k").agg(
        n_topics=("topic_label", "count"),
        n_groups=("group_id", "nunique"),
        ships=("ship", lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()
    mo.vstack([mo.md("### Topics per selected k"), mo.ui.table(_summary)])
    return


@app.cell
def umap_controls(mo):
    n_neighbors_slider = mo.ui.slider(
        start=5, stop=50, value=5, step=5, label="n_neighbors"
    )
    min_dist_slider = mo.ui.slider(
        start=0.0, stop=1.0, value=0.1, step=0.05, label="min_dist"
    )
    metric_dropdown = mo.ui.dropdown(
        options=["cosine", "euclidean", "correlation", "manhattan"],
        value="cosine",
        label="metric",
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
    topic_meta,
    umap,
):
    _reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_slider.value,
        min_dist=min_dist_slider.value,
        metric=metric_dropdown.value,
        random_state=42,
    )
    _embedding = _reducer.fit_transform(topic_features.values)

    umap_df = topic_meta.copy()
    umap_df["umap_1"] = _embedding[:, 0]
    umap_df["umap_2"] = _embedding[:, 1]

    _top3_with_weights = topic_features.apply(
        lambda row: ", ".join(
            f"{sp} ({row[sp]:.2f})" for sp in row.nlargest(3).index
        ),
        axis=1,
    )
    umap_df["top3_detail"] = _top3_with_weights.values

    mo.md(
        f"UMAP done: n_neighbors={n_neighbors_slider.value}, "
        f"min_dist={min_dist_slider.value}, metric={metric_dropdown.value}"
    )
    return (umap_df,)


@app.cell(hide_code=True)
def color_controls(mo, topic_features):
    color_by = mo.ui.dropdown(
        options=[
            "dominant_species", "ship", "best_k",
            "component", "n_segments", "n_tows_merged",
            "year", "species_weight",
        ],
        value="dominant_species",
        label="Color by",
    )

    # Top 10 species by total weight across all topics
    _species_sorted = (
        topic_features.sum(axis=0).sort_values(ascending=False).head(10).index.tolist()
    )
    species_dropdown = mo.ui.dropdown(
        options=_species_sorted,
        value=_species_sorted[0] if _species_sorted else None,
        label="Species (top 10, used when Color by = species_weight)",
        searchable=True,
    )
    mo.md("### UMAP Coloring")
    mo.hstack([color_by, species_dropdown])
    return color_by, species_dropdown


@app.cell(hide_code=True)
def umap_scatter(color_by, mo, px, species_dropdown, topic_features, umap_df):
    _df = umap_df.copy()

    if color_by.value == "species_weight" and species_dropdown.value is not None:
        _sp = species_dropdown.value
        _df["_species_weight"] = topic_features[_sp].values
        _color_col = "_species_weight"
        _title = (
            f"UMAP of NMF topics \u2014 colored by weight of \u201c{_sp}\u201d  "
            "(select points to see segments on map)"
        )
        _color_scale = "Blues"
    else:
        _color_col = color_by.value
        _title = (
            f"UMAP of NMF topics \u2014 colored by {_color_col}  "
            "(select points to see segments on map)"
        )
        _color_scale = "Blues"

    _fig = px.scatter(
        _df,
        x="umap_1", y="umap_2",
        color=_color_col,
        color_continuous_scale=_color_scale,
        hover_data=[
            "topic_label", "ship", "dominant_species",
            "top3_detail", "best_k", "n_segments",
            "year", "period_start", "period_end",
            "group_id", "component",
        ],
        title=_title,
        width=1000, height=650,
    )
    _fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color="white")))
    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font=dict(family="monospace"),
        legend=dict(font=dict(size=10), itemsizing="constant"),
        dragmode="select",
    )
    if _color_col == "_species_weight":
        _fig.update_layout(
            coloraxis_colorbar=dict(title=f"{species_dropdown.value}<br>weight"),
        )
    umap_plot = mo.ui.plotly(_fig)
    return (umap_plot,)


@app.cell
def _(umap_plot):
    umap_plot
    return


@app.cell
def _(go, mo, np, umap_plot):
    _selected = umap_plot.value or []

    import re
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list
    from plotly.subplots import make_subplots

    _topics = []
    for _pt in _selected:
        _sw = {}
        for _m in re.finditer(r"([^,]+?)\s*\((\d+\.\d+)\)", _pt.get("top3_detail", "")):
            _sw[_m.group(1).strip()] = float(_m.group(2))
        if _sw:
            _topics.append({"label": _pt.get("topic_label", "?"), "species": _sw})

    if not _topics:
        _out = mo.md("*Select topics in the UMAP scatter to see species barplots.*")
    else:
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
        _ncols = min(7, _n)
        _nrows = -(-_n // _ncols)

        _fig = make_subplots(
            rows=_nrows, cols=_ncols,
            subplot_titles=[_topics[i]["label"] for i in _order],
            vertical_spacing=max(0.04, 0.15 / _nrows),
            horizontal_spacing=0.06,
        )

        _palette = [
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        ]
        _sp_colors = {sp: _palette[i % len(_palette)] for i, sp in enumerate(_all_sp)}

        for _pos, _idx in enumerate(_order):
            _t = _topics[_idx]
            _sps = list(_t["species"].keys())
            _ws = list(_t["species"].values())
            _fig.add_trace(
                go.Bar(
                    x=_sps, y=_ws,
                    marker_color=[_sp_colors[s] for s in _sps],
                    showlegend=False,
                    text=[f"{w:.2f}" for w in _ws],
                    textposition="outside",
                ),
                row=_pos // _ncols + 1, col=_pos % _ncols + 1,
            )
            _fig.update_yaxes(range=[0, 1], row=_pos // _ncols + 1, col=_pos % _ncols + 1)

        _fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=250 * _nrows, width=280 * _ncols,
            font=dict(family="monospace", size=9),
            title="Top-3 species per selected topic (ordered by similarity)",
            margin=dict(t=60, b=20),
        )
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def dominant_species_breakdown(mo, topic_meta):
    _counts = topic_meta["dominant_species"].value_counts().head(20).reset_index()
    _counts.columns = ["species", "n_topics"]
    mo.vstack([mo.md("### Top 20 dominant species across all topics"), mo.ui.table(_counts)])
    return


@app.cell(hide_code=True)
def species_heatmap():
    # _top_species = topic_features.sum(axis=0).nlargest(30).index.tolist()
    # _heat_data = topic_features[_top_species].copy()
    # _heat_data.index = topic_meta["topic_label"].values

    # _fig = px.imshow(
    #     _heat_data.values,
    #     x=_top_species,
    #     y=_heat_data.index.tolist(),
    #     color_continuous_scale="Viridis",
    #     aspect="auto",
    #     title="Top 30 species weights across all topics (H_normalized)",
    #     width=1100,
    #     height=max(400, len(_heat_data) * 3),
    # )
    # _fig.update_layout(
    #     template="plotly_dark",
    #     paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
    #     font=dict(family="monospace", size=9),
    #     xaxis=dict(tickangle=45),
    # )
    # mo.ui.plotly(_fig)
    return


@app.cell
def clustering_imports():
    from fast_hdbscan import HDBSCAN as FastHDBSCAN

    return (FastHDBSCAN,)


@app.cell
def hdbscan_controls(mo):
    hdb_min_cluster = mo.ui.slider(
        start=3, stop=20, value=5, step=1, label="min_cluster_size"
    )
    hdb_method = mo.ui.dropdown(
        options=["eom", "leaf"], value="eom", label="method"
    )
    hdb_space = mo.ui.dropdown(
        options=["high_dim", "umap_2d"], value="high_dim", label="space"
    )
    mo.md("### HDBSCAN Clustering")
    mo.hstack([hdb_min_cluster, hdb_method, hdb_space])
    return hdb_method, hdb_min_cluster, hdb_space


@app.cell
def run_hdbscan(
    FastHDBSCAN,
    hdb_method,
    hdb_min_cluster,
    hdb_space,
    mo,
    topic_features,
    umap_df,
):
    if hdb_space.value == "high_dim":
        _data = topic_features.values
        _metric = "cosine"
    else:
        _data = umap_df[["umap_1", "umap_2"]].values
        _metric = "euclidean"

    _hdb = FastHDBSCAN(
        min_cluster_size=hdb_min_cluster.value,
        cluster_selection_method=hdb_method.value,
        metric=_metric,
    )
    _hdb.fit(_data)

    hdb_labels = _hdb.labels_
    hdb_probs = _hdb.probabilities_

    _n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    _n_noise = int((hdb_labels == -1).sum())

    mo.md(
        f"HDBSCAN: **{_n_clusters} clusters**, **{_n_noise} noise** "
        f"({hdb_space.value}, {hdb_method.value}, min_cluster_size={hdb_min_cluster.value})"
    )
    return (hdb_labels,)


@app.cell
def hdbscan_scatter(go, hdb_labels, mo, umap_df):
    _df = umap_df.copy()
    _df["cluster"] = hdb_labels.astype(str)

    _fig = go.Figure()
    for _cl in sorted(_df["cluster"].unique(), key=lambda x: int(x)):
        _sub = _df[_df["cluster"] == _cl]
        _is_noise = _cl == "-1"
        _fig.add_trace(go.Scatter(
            x=_sub["umap_1"], y=_sub["umap_2"],
            mode="markers",
            name="Noise" if _is_noise else f"Cluster {_cl}",
            marker=dict(
                size=4 if _is_noise else 8,
                opacity=0.25 if _is_noise else 0.85,
                line=dict(width=0.3, color="white"),
            ),
            text=_sub["topic_label"] + "<br>" + _sub["dominant_species"],
            hoverinfo="text+name",
        ))
    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        title="HDBSCAN Clusters on UMAP",
        width=900, height=600,
        font=dict(family="monospace"),
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def load_kmst_graph(Path, json, mo):
    _graph_path = Path("data/nmf_per_tow/topic_kmst_graph.json")
    with open(_graph_path) as _f:
        kmst_data = json.load(_f)

    for _n in kmst_data["nodes"]:
        _t3 = _n["top3_species"]
        _n["top3_detail"] = ", ".join(
            f"{sp} ({w:.2f})" for sp, w in sorted(_t3.items(), key=lambda x: -x[1])
        )

    mo.md(
        f"### k-MST Graph (pre-computed)\n"
        f"**{kmst_data['n_nodes']} nodes**, **{kmst_data['n_edges']} edges** "
        f"(metric={kmst_data['params']['metric']}, k={kmst_data['params']['num_neighbors']}, "
        f"eps={kmst_data['params']['epsilon']})"
    )
    return (kmst_data,)


@app.cell
def compute_force_layout(kmst_data, np):
    import networkx as nx

    _G = nx.Graph()
    _G.add_nodes_from(range(kmst_data["n_nodes"]))
    for _e in kmst_data["links"]:
        _G.add_edge(_e["source"], _e["target"], weight=_e["weight"])

    _pos = nx.spring_layout(_G, k=0.3, iterations=100, seed=42, weight="weight")

    graph_xs = np.array([_pos[i][0] for i in range(kmst_data["n_nodes"])])
    graph_ys = np.array([_pos[i][1] for i in range(kmst_data["n_nodes"])])

    _coo_sources = np.array([_e["source"] for _e in kmst_data["links"]])
    _coo_targets = np.array([_e["target"] for _e in kmst_data["links"]])
    graph_sources = _coo_sources
    graph_targets = _coo_targets
    return graph_sources, graph_targets, graph_xs, graph_ys


@app.cell
def graph_color_controls(mo):
    graph_color_by = mo.ui.dropdown(
        options=["cluster", "dominant_species", "ship", "best_k", "group_id"],
        value="cluster",
        label="Color by",
    )
    graph_color_by
    return (graph_color_by,)


@app.cell(hide_code=True)
def force_graph(
    go,
    graph_color_by,
    graph_sources,
    graph_targets,
    graph_xs,
    graph_ys,
    hdb_labels,
    kmst_data,
    mo,
):
    _edge_x, _edge_y = [], []
    for _s, _t in zip(graph_sources, graph_targets):
        _edge_x += [float(graph_xs[_s]), float(graph_xs[_t]), None]
        _edge_y += [float(graph_ys[_s]), float(graph_ys[_t]), None]

    _col_name = graph_color_by.value
    if _col_name == "cluster":
        _color_vals = [int(hdb_labels[i]) for i in range(len(kmst_data["nodes"]))]
    else:
        _color_vals = [_n[_col_name] for _n in kmst_data["nodes"]]

    _hover = [
        f"{_n['label']}<br>Cluster: {int(hdb_labels[_n['id']])}<br>"
        f"Species: {_n['dominant_species']}<br>Ship: {_n['ship']}<br>"
        f"k={_n['best_k']}<br>{_n['period_start']} — {_n['period_end']}"
        for _n in kmst_data["nodes"]
    ]

    _fig = go.Figure()

    _fig.add_trace(go.Scattergl(
        x=_edge_x, y=_edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(100,100,100,0.5)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    _is_categorical = isinstance(_color_vals[0], str)
    if _is_categorical:
        _cats = sorted(set(_color_vals))
        _palette = [
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for _ci, _cat in enumerate(_cats):
            _mask = [_j for _j, _v in enumerate(_color_vals) if _v == _cat]
            _fig.add_trace(go.Scattergl(
                x=graph_xs[_mask], y=graph_ys[_mask],
                mode="markers",
                name=str(_cat),
                marker=dict(
                    size=5, color=_palette[_ci % len(_palette)],
                    opacity=0.8, line=dict(width=0, color="white"),
                ),
                text=[_hover[_j] for _j in _mask],
                hoverinfo="text",
            ))
    else:
        _fig.add_trace(go.Scattergl(
            x=graph_xs.tolist(), y=graph_ys.tolist(),
            mode="markers",
            marker=dict(
                size=5, color=_color_vals,
                colorscale="Viridis", showscale=True, opacity=0.8,
                colorbar=dict(title=_col_name, thickness=12, len=0.5),
                line=dict(width=0),
            ),
            text=_hover,
            hoverinfo="text",
            showlegend=False,
        ))

    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        title=f"k-MST Force Layout — colored by {_col_name}",
        width=1000, height=750,
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        font=dict(family="monospace"),
        legend=dict(font=dict(size=9), itemsizing="constant"),
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def kmst_umap_scatter(go, hdb_labels, kmst_data, mo, pd, px):
    _has_umap = "umap_x" in kmst_data["nodes"][0]

    if not _has_umap:
        kmst_umap_plot = mo.md(
            "*Re-run `build_topic_graph.py` in the kMST environment to save UMAP coords.*"
        )
    else:
        _df = pd.DataFrame([
            {
                "umap_x": _n["umap_x"],
                "umap_y": _n["umap_y"],
                "topic_label": _n["label"],
                "dominant_species": _n["dominant_species"],
                "ship": _n["ship"],
                "best_k": _n["best_k"],
                "group_id": _n["group_id"],
                "component": _n["component"],
                "period_start": _n["period_start"],
                "period_end": _n["period_end"],
                "top3_detail": _n["top3_detail"],
                "cluster": str(int(hdb_labels[_n["id"]])),
            }
            for _n in kmst_data["nodes"]
        ])

        _fig = px.scatter(
            _df,
            x="umap_x", y="umap_y",
            color="dominant_species",
            hover_data=[
                "topic_label", "dominant_species", "ship",
                "top3_detail", "best_k", "group_id",
                "component", "period_start", "period_end", "cluster",
            ],
            title="k-MST UMAP \u2014 colored by dominant species (select for barplot)",
            width=1000, height=700,
        )

        _edge_x, _edge_y = [], []
        _xs, _ys = _df["umap_x"].values, _df["umap_y"].values
        for _e in kmst_data["links"]:
            _s, _t = _e["source"], _e["target"]
            _edge_x += [float(_xs[_s]), float(_xs[_t]), None]
            _edge_y += [float(_ys[_s]), float(_ys[_t]), None]

        _fig.add_trace(go.Scattergl(
            x=_edge_x, y=_edge_y,
            mode="lines",
            line=dict(width=0.4, color="rgba(150,150,150,0.3)"),
            hoverinfo="skip",
            showlegend=False,
        ))
        _fig.data = (_fig.data[-1],) + _fig.data[:-1]

        _fig.update_traces(
            marker=dict(size=6, opacity=0.85, line=dict(width=0.3, color="white")),
            selector=dict(mode="markers"),
        )
        _fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font=dict(family="monospace"),
            legend=dict(font=dict(size=9), itemsizing="constant"),
            xaxis=dict(visible=False, scaleanchor="y"),
            yaxis=dict(visible=False),
            dragmode="select",
        )
        kmst_umap_plot = mo.ui.plotly(_fig)

    kmst_umap_plot
    return (kmst_umap_plot,)


@app.cell
def linked_section_header(mo):
    mo.md("""
    ## Linked KMST UMAP → NMF UMAP + Small Multiples
    "
        "Select topics in the **k-MST UMAP** above. The same topics are highlighted "
        "in the original **NMF UMAP** below, and per-topic species barplots are shown underneath.
    """)
    return


@app.cell
def _(kmst_umap_plot):
    kmst_umap_plot
    return


@app.cell(hide_code=True)
def linked_umap_highlight(go, kmst_umap_plot, mo, umap_df):
    _selected = kmst_umap_plot.value or []
    _sel_labels = {_pt.get("topic_label") for _pt in _selected if "topic_label" in _pt}

    _df = umap_df.copy()
    _df["_is_selected"] = _df["topic_label"].isin(_sel_labels)

    _bg = _df[~_df["_is_selected"]]
    _fg = _df[_df["_is_selected"]]

    _fig = go.Figure()
    _fig.add_trace(go.Scattergl(
        x=_bg["umap_1"], y=_bg["umap_2"],
        mode="markers",
        marker=dict(
            size=5,
            color="rgba(150,150,150,0.25)" if len(_sel_labels) else "rgba(108,140,255,0.6)",
            line=dict(width=0),
        ),
        text=_bg["topic_label"] + "<br>" + _bg["dominant_species"],
        hoverinfo="text",
        name="other topics" if len(_sel_labels) else "all topics",
        showlegend=False,
    ))
    if len(_fg):
        _fig.add_trace(go.Scattergl(
            x=_fg["umap_1"], y=_fg["umap_2"],
            mode="markers",
            marker=dict(
                size=11,
                color="#f28e2b",
                line=dict(width=1, color="white"),
                symbol="circle",
            ),
            text=_fg["topic_label"] + "<br>" + _fg["dominant_species"]
                 + "<br>" + _fg["top3_detail"],
            hoverinfo="text",
            name="selected (from KMST)",
            showlegend=True,
        ))

    _title = (
        f"NMF UMAP \u2014 {len(_fg)} topic(s) highlighted from KMST selection"
        if len(_sel_labels) else
        "NMF UMAP \u2014 select topics in the KMST UMAP above to highlight"
    )
    _fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        title=_title,
        width=1000, height=600,
        font=dict(family="monospace"),
        legend=dict(font=dict(size=10), itemsizing="constant"),
    )
    umap_umap = mo.ui.plotly(_fig)
    return (umap_umap,)


@app.cell(hide_code=True)
def linked_small_multiples(go, kmst_umap_plot, mo, np):
    _selected = kmst_umap_plot.value or []

    import re as _re
    from scipy.spatial.distance import pdist as _pdist
    from scipy.cluster.hierarchy import linkage as _linkage, leaves_list as _leaves_list
    from plotly.subplots import make_subplots as _make_subplots

    _topics = []
    for _pt in _selected:
        _sw = {}
        for _m in _re.finditer(r"([^,]+?)\s*\((\d+\.\d+)\)", _pt.get("top3_detail", "")):
            _sw[_m.group(1).strip()] = float(_m.group(2))
        if _sw:
            _topics.append({"label": _pt.get("topic_label", "?"), "species": _sw})

    if not _topics:
        sm = mo.md("*Select topics in the k-MST UMAP to see species barplots.*")
    else:
        _all_sp = sorted({sp for t in _topics for sp in t["species"]})
        _sp_idx = {sp: i for i, sp in enumerate(_all_sp)}
        _vecs = np.zeros((len(_topics), len(_all_sp)))
        for _i, _t in enumerate(_topics):
            for _sp, _w in _t["species"].items():
                _vecs[_i, _sp_idx[_sp]] = _w

        if len(_topics) > 2:
            _d = np.nan_to_num(_pdist(_vecs, metric="cosine"), nan=1.0)
            _order = _leaves_list(_linkage(_d, method="average")).tolist()
        else:
            _order = list(range(len(_topics)))

        _n = len(_topics)
        _ncols = min(7, _n)
        _nrows = -(-_n // _ncols)

        _fig = _make_subplots(
            rows=_nrows, cols=_ncols,
            subplot_titles=[_topics[i]["label"] for i in _order],
            vertical_spacing=max(0.04, 0.15 / _nrows),
            horizontal_spacing=0.06,
        )

        _palette = [
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        ]
        _sp_colors = {sp: _palette[i % len(_palette)] for i, sp in enumerate(_all_sp)}
        _shown_sp = set()

        for _pos, _idx in enumerate(_order):
            _t = _topics[_idx]
            _sps = list(_t["species"].keys())
            _ws = list(_t["species"].values())
            for _si in range(len(_sps)):
                _sp = _sps[_si]
                _w = _ws[_si]
                _show = _sp not in _shown_sp
                _fig.add_trace(
                    go.Bar(
                        x=[_si], y=[_w],
                        marker_color=_sp_colors[_sp],
                        name=_sp,
                        legendgroup=_sp,
                        showlegend=_show,
                        text=[f"{_w:.2f}"],
                        textposition="outside",
                        width=0.6,
                    ),
                    row=_pos // _ncols + 1, col=_pos % _ncols + 1,
                )
                if _show:
                    _shown_sp.add(_sp)
            _fig.update_xaxes(showticklabels=False, row=_pos // _ncols + 1, col=_pos % _ncols + 1)
            _fig.update_yaxes(range=[0, 1], row=_pos // _ncols + 1, col=_pos % _ncols + 1)

        _fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            height=220 * _nrows + 40, width=280 * _ncols + 160,
            font=dict(family="monospace", size=9),
            title="Top-3 species per selected topic (ordered by similarity)",
            margin=dict(t=60, b=20, r=160),
            legend=dict(
                title="Species", font=dict(size=10),
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            ),
            barmode="group",
        )
        sm = mo.ui.plotly(_fig)
    return (sm,)


@app.cell
def _(kmst_umap_plot, mo, sm, umap_umap):
    mo.vstack([mo.hstack([kmst_umap_plot,umap_umap]),sm])
    return


if __name__ == "__main__":
    app.run()
