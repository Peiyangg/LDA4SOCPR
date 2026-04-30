import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import umap
    import plotly.graph_objects as go
    import kmapper as km
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    import networkx as nx
    import h3

    return SklearnHDBSCAN, go, h3, km, mo, np, nx, pd, umap


@app.cell
def _(pd):
    hex_r3 = pd.read_csv("data/hex_features_res3.csv")
    hex_r3_meta = pd.read_csv("data/hex_metadata_res3.csv")
    hex_r3_meta.index = (
        hex_r3_meta["h3_cell"].astype(str) + "_" + hex_r3_meta["year"].astype(str)
    )
    hex_r3.index = (
        hex_r3["h3_cell"].astype(str) + "_" + hex_r3["year"].astype(str)
    )
    hex_r3_clean = hex_r3.drop(columns=["h3_cell", "year"])
    return hex_r3_clean, hex_r3_meta


@app.cell
def _(hex_r3_clean, mo, umap):
    with mo.status.spinner("Computing UMAP projection (Bray-Curtis)..."):
        _reducer = umap.UMAP(
            metric="braycurtis", n_components=2, n_neighbors=5, random_state=42
        )
        umap_emb = _reducer.fit_transform(hex_r3_clean)
    return (umap_emb,)


@app.cell(hide_code=True)
def _(np):
    from sklearn.base import BaseEstimator, ClusterMixin

    class FLASCClusterer(BaseEstimator, ClusterMixin):
        """Wraps fast_hdbscan HDBSCAN + BranchDetector for use with KeplerMapper."""

        def __init__(self, min_cluster_size=10, metric="braycurtis"):
            self.min_cluster_size = min_cluster_size
            self.metric = metric

        def fit(self, X, y=None):
            self.labels_ = self.fit_predict(X)
            return self

        def fit_predict(self, X, y=None):
            from fast_hdbscan import HDBSCAN as FastHDBSCAN
            from fast_hdbscan import BranchDetector

            if len(X) < self.min_cluster_size * 2:
                return np.full(len(X), -1)
            try:
                hdb = FastHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric=self.metric,
                    allow_single_cluster=True,
                )
                hdb.fit(X)
                if np.all(hdb.labels_ == -1):
                    return hdb.labels_
                bd = BranchDetector()
                bd.fit(hdb)
                return bd.labels_
            except Exception:
                return np.full(len(X), -1)

    class PLSCANClusterer(BaseEstimator, ClusterMixin):
        """Wraps fast_hdbscan PLSCAN for use with KeplerMapper."""

        def __init__(self, min_samples=5, metric="braycurtis"):
            self.min_samples = min_samples
            self.metric = metric

        def fit(self, X, y=None):
            self.labels_ = self.fit_predict(X)
            return self

        def fit_predict(self, X, y=None):
            from fast_hdbscan import PLSCAN

            if len(X) < self.min_samples * 3:
                return np.full(len(X), -1)
            try:
                plscan = PLSCAN(min_samples=self.min_samples, metric=self.metric)
                return plscan.fit_predict(X)
            except Exception:
                return np.full(len(X), -1)

    return FLASCClusterer, PLSCANClusterer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapper Parameters
    """)
    return


@app.cell
def _(mo):
    min_cluster_size_slider = mo.ui.slider(
        start=3, stop=30, step=1, value=5, label="min_cluster_size"
    )
    n_cubes_slider = mo.ui.slider(
        start=3, stop=25, step=1, value=10, label="n_cubes"
    )
    perc_overlap_slider = mo.ui.slider(
        start=0.1, stop=0.7, step=0.05, value=0.5, label="perc_overlap"
    )
    mo.vstack([min_cluster_size_slider, n_cubes_slider, perc_overlap_slider])
    return min_cluster_size_slider, n_cubes_slider, perc_overlap_slider


@app.cell(hide_code=True)
def _(
    FLASCClusterer,
    PLSCANClusterer,
    SklearnHDBSCAN,
    hex_r3_clean,
    km,
    min_cluster_size_slider,
    mo,
    n_cubes_slider,
    perc_overlap_slider,
    umap_emb,
):
    _mcs = min_cluster_size_slider.value
    _n = n_cubes_slider.value
    _p = perc_overlap_slider.value

    _clusterers = {
        "HDBSCAN (EOM)": SklearnHDBSCAN(
            metric="braycurtis",
            cluster_selection_method="eom",
            min_cluster_size=_mcs,
            allow_single_cluster=True,
        ),
        "HDBSCAN (Leaf)": SklearnHDBSCAN(
            metric="braycurtis",
            cluster_selection_method="leaf",
            min_cluster_size=_mcs,
            allow_single_cluster=True,
        ),
        "FLASC": FLASCClusterer(min_cluster_size=_mcs),
        "PLSCAN": PLSCANClusterer(min_samples=_mcs),
    }

    _mapper = km.KeplerMapper(verbose=0)
    _cover = km.Cover(n_cubes=_n, perc_overlap=_p)
    _X = hex_r3_clean.values

    with mo.status.spinner("Running Mapper with 4 clustering variants..."):
        mapper_results = {}
        for _name, _clf in _clusterers.items():
            mapper_results[_name] = _mapper.map(
                umap_emb, _X, clusterer=_clf, cover=_cover
            )
    return (mapper_results,)


@app.cell(hide_code=True)
def _(mapper_results, np, nx):
    variant_names = list(mapper_results.keys())

    all_graph_data = {}
    for _name in variant_names:
        _graph = mapper_results[_name]
        _node_ids = list(_graph["nodes"].keys())
        _node_members = [_graph["nodes"][nid] for nid in _node_ids]
        _id_to_idx = {nid: i for i, nid in enumerate(_node_ids)}

        _G = nx.Graph()
        for _i in range(len(_node_ids)):
            _G.add_node(_i)

        for _nid in _graph.get("links", {}):
            if _nid in _id_to_idx:
                _src = _id_to_idx[_nid]
                for _linked in _graph["links"][_nid]:
                    if _linked in _id_to_idx:
                        _tgt = _id_to_idx[_linked]
                        _G.add_edge(_src, _tgt)

        if len(_G) > 0:
            _pos = nx.spring_layout(
                _G, seed=42, k=2.0 / np.sqrt(len(_G)), iterations=100
            )
        else:
            _pos = {}

        _node_info = []
        for _i, _nid in enumerate(_node_ids):
            _members = _node_members[_i]
            _node_info.append(
                {
                    "x": _pos.get(_i, (0, 0))[0],
                    "y": _pos.get(_i, (0, 0))[1],
                    "members": _members,
                    "size": len(_members),
                }
            )

        all_graph_data[_name] = {
            "node_info": _node_info,
            "edges": list(_G.edges()),
            "n_nodes": len(_node_ids),
            "n_edges": _G.number_of_edges(),
        }
    return all_graph_data, variant_names


@app.cell(hide_code=True)
def _():
    ZONE_ORDER = ["North of STF", "STF-SAF", "SAF-PF", "PF-SACCF", "SACCF-SBDY", "South of SBDY"]
    ZONE_COLORS = {
        "North of STF": "#e74c3c",
        "STF-SAF": "#e67e22",
        "SAF-PF": "#2ecc71",
        "PF-SACCF": "#3498db",
        "SACCF-SBDY": "#9b59b6",
        "South of SBDY": "#1a1a2e",
    }
    CATEGORICAL_COLS = {"ocean_zone"}
    return CATEGORICAL_COLS, ZONE_COLORS, ZONE_ORDER


@app.cell(hide_code=True)
def _(CATEGORICAL_COLS, ZONE_COLORS, ZONE_ORDER, go, np, pd):
    def make_mapper_fig(graph_data, title, meta_df, color_col="center_lat", height=400, interactive=False):
        node_info = graph_data["node_info"]
        edges = graph_data["edges"]
        is_cat = color_col in CATEGORICAL_COLS

        if not node_info:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (empty)", height=height)
            return fig

        edge_x, edge_y = [], []
        for u, v in edges:
            edge_x.extend([node_info[u]["x"], node_info[v]["x"], None])
            edge_y.extend([node_info[u]["y"], node_info[v]["y"], None])

        sizes = [n["size"] for n in node_info]
        max_s = max(sizes) if sizes else 1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="#ccc"),
                hoverinfo="none", showlegend=False,
            )
        )

        if is_cat:
            node_labels = []
            for n in node_info:
                vals = meta_df.iloc[n["members"]][color_col]
                node_labels.append(vals.mode().iloc[0] if len(vals) > 0 else ZONE_ORDER[0])

            for zone in ZONE_ORDER:
                _mask = [l == zone for l in node_labels]
                if not any(_mask):
                    continue
                _ni = [node_info[i] for i, m in enumerate(_mask) if m]
                _sz = [sizes[i] for i, m in enumerate(_mask) if m]
                _idx = [i for i, m in enumerate(_mask) if m]
                fig.add_trace(
                    go.Scatter(
                        x=[n["x"] for n in _ni],
                        y=[n["y"] for n in _ni],
                        mode="markers",
                        marker=dict(
                            size=[max(10, s / max_s * 35) for s in _sz],
                            color=ZONE_COLORS[zone],
                            line=dict(width=1, color="black"),
                        ),
                        text=[f"Node {i}: {node_info[i]['size']} pts<br>{color_col}: {zone}" for i in _idx],
                        hoverinfo="text",
                        customdata=_idx,
                        name=zone,
                        showlegend=True,
                    )
                )
        else:
            node_colors = []
            for n in node_info:
                vals = meta_df.iloc[n["members"]][color_col].values
                node_colors.append(float(np.nanmean(vals)) if len(vals) > 0 else 0)
            fig.add_trace(
                go.Scatter(
                    x=[n["x"] for n in node_info],
                    y=[n["y"] for n in node_info],
                    mode="markers",
                    marker=dict(
                        size=[max(10, s / max_s * 35) for s in sizes],
                        color=node_colors,
                        colorscale="Viridis",
                        colorbar=dict(title=color_col, len=0.7),
                        line=dict(width=1, color="black"),
                    ),
                    text=[
                        f"Node {i}: {n['size']} pts<br>{color_col}: {node_colors[i]:.2f}"
                        for i, n in enumerate(node_info)
                    ],
                    hoverinfo="text",
                    customdata=list(range(len(node_info))),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=-0.05) if is_cat else {},
        )
        if interactive:
            fig.update_layout(clickmode="event+select", dragmode="lasso")
        return fig

    return (make_mapper_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapper Comparison
    """)
    return


@app.cell
def _(all_graph_data, hex_r3_meta, make_mapper_fig, mo, variant_names):
    _figs = [
        make_mapper_fig(all_graph_data[n], n, meta_df=hex_r3_meta, height=350) for n in variant_names
    ]
    mo.vstack(
        [
            mo.hstack(_figs[:2], widths="equal"),
            mo.hstack(_figs[2:], widths="equal"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(all_graph_data, hex_r3_clean, mo, np, pd, variant_names):
    _total = len(hex_r3_clean)
    _rows = []
    for _name in variant_names:
        _gd = all_graph_data[_name]
        _ni = _gd["node_info"]
        _sizes = [n["size"] for n in _ni]
        _all_members = set()
        for n in _ni:
            _all_members.update(n["members"])
        _rows.append(
            {
                "Method": _name,
                "Nodes": _gd["n_nodes"],
                "Edges": _gd["n_edges"],
                "Avg node size": round(np.mean(_sizes), 1) if _sizes else 0,
                "Max node size": max(_sizes) if _sizes else 0,
                "Coverage": f"{len(_all_members)}/{_total} ({100 * len(_all_members) / _total:.0f}%)",
            }
        )
    mo.ui.table(pd.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Explorer

    Use **lasso** or **box select** on the Mapper graph to select nodes.
    The UMAP plot highlights the corresponding data points.
    """)
    return


@app.cell
def _(CATEGORICAL_COLS, hex_r3_meta, mo, variant_names):
    _color_cols = [
        c for c in hex_r3_meta.columns
        if (hex_r3_meta[c].dtype in ("float64", "int64") and c not in ("n_tows", "n_ships"))
        or c in CATEGORICAL_COLS
    ]
    variant_dropdown = mo.ui.dropdown(
        options=variant_names,
        value=variant_names[0],
        label="Clustering method",
    )
    color_dropdown = mo.ui.dropdown(
        options=_color_cols,
        value="center_lat",
        label="Color by",
    )
    mo.hstack([variant_dropdown, color_dropdown], justify="start", gap=1)
    return color_dropdown, variant_dropdown


@app.cell
def _(
    all_graph_data,
    color_dropdown,
    hex_r3_meta,
    make_mapper_fig,
    mo,
    variant_dropdown,
):
    _fig = make_mapper_fig(
        all_graph_data[variant_dropdown.value],
        variant_dropdown.value,
        meta_df=hex_r3_meta,
        color_col=color_dropdown.value,
        height=500,
        interactive=True,
    )
    mapper_plot = mo.ui.plotly(_fig)
    return (mapper_plot,)


@app.cell(hide_code=True)
def _(
    CATEGORICAL_COLS,
    ZONE_COLORS,
    ZONE_ORDER,
    all_graph_data,
    color_dropdown,
    go,
    hex_r3_meta,
    mapper_plot,
    np,
    umap_emb,
    variant_dropdown,
):
    _node_info = all_graph_data[variant_dropdown.value]["node_info"]
    _color_col = color_dropdown.value
    _is_cat = _color_col in CATEGORICAL_COLS
    selected_indices = set()

    _val = mapper_plot.value
    _points = _val if isinstance(_val, list) else (_val.get("points", []) if isinstance(_val, dict) else [])
    for _pt in _points:
        _cn = _pt.get("curveNumber", _pt.get("curve_number"))
        if _cn == 0:
            continue
        _cd = _pt.get("customdata")
        if _cd is not None:
            _nidx = _cd if isinstance(_cd, int) else _cd[0] if isinstance(_cd, list) else None
            if _nidx is not None and 0 <= _nidx < len(_node_info):
                selected_indices.update(_node_info[_nidx]["members"])

    umap_fig = go.Figure()

    def _add_cat_traces(idx_array, size, opacity=0.7):
        _labels = hex_r3_meta.iloc[idx_array][_color_col].values
        for zone in ZONE_ORDER:
            _zmask = _labels == zone
            if not _zmask.any():
                continue
            _zi = idx_array[_zmask]
            umap_fig.add_trace(
                go.Scatter(
                    x=umap_emb[_zi, 0], y=umap_emb[_zi, 1],
                    mode="markers",
                    marker=dict(size=size, color=ZONE_COLORS[zone], opacity=opacity,
                                line=dict(width=0.5, color="black") if size > 4 else {}),
                    text=hex_r3_meta.iloc[_zi].index.tolist(),
                    hoverinfo="text", name=zone, showlegend=True,
                )
            )

    if selected_indices:
        _all_idx = np.arange(len(umap_emb))
        _fg_idx = np.array(sorted(selected_indices))
        _bg_mask = ~np.isin(_all_idx, _fg_idx)

        umap_fig.add_trace(
            go.Scatter(
                x=umap_emb[_bg_mask, 0], y=umap_emb[_bg_mask, 1],
                mode="markers",
                marker=dict(size=3, color="lightgray", opacity=0.25),
                hoverinfo="skip", showlegend=False,
            )
        )
        if _is_cat:
            _add_cat_traces(_fg_idx, size=7)
        else:
            umap_fig.add_trace(
                go.Scatter(
                    x=umap_emb[_fg_idx, 0], y=umap_emb[_fg_idx, 1],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=hex_r3_meta.iloc[_fg_idx][_color_col].values,
                        colorscale="Viridis",
                        colorbar=dict(title=_color_col),
                        line=dict(width=0.5, color="black"),
                    ),
                    text=hex_r3_meta.iloc[_fg_idx].index.tolist(),
                    hoverinfo="text", showlegend=False,
                )
            )
        umap_fig.update_layout(title=f"UMAP — {len(_fg_idx)} points selected")
    else:
        if _is_cat:
            _add_cat_traces(np.arange(len(umap_emb)), size=4)
        else:
            umap_fig.add_trace(
                go.Scatter(
                    x=umap_emb[:, 0], y=umap_emb[:, 1],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=hex_r3_meta[_color_col],
                        colorscale="Viridis",
                        colorbar=dict(title=_color_col),
                        opacity=0.7,
                    ),
                    text=hex_r3_meta.index.tolist(),
                    hoverinfo="text", showlegend=False,
                )
            )
        umap_fig.update_layout(title="UMAP Projection (Bray-Curtis)")

    umap_fig.update_layout(legend=dict(orientation="h", y=-0.05) if _is_cat else {})
    return selected_indices, umap_fig


@app.cell(hide_code=True)
def _(h3, hex_r3_meta):
    import json
    from pyproj import Transformer
    import cartopy.io.shapereader as shpreader

    _proj = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    def _project_boundary(cell):
        boundary = h3.cell_to_boundary(cell)
        xs, ys = [], []
        for lat, lon in boundary:
            x, y = _proj.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        xs.append(xs[0])
        ys.append(ys[0])
        return xs, ys

    hex_boundaries_proj = {}
    for _cell in hex_r3_meta["h3_cell"].unique():
        hex_boundaries_proj[_cell] = _project_boundary(_cell)

    _land_shp = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    _reader = shpreader.Reader(_land_shp)
    coast_xs, coast_ys = [], []
    for _geom in _reader.geometries():
        if _geom.centroid.y > -30:
            continue
        if _geom.geom_type == "Polygon":
            _polys = [_geom]
        elif _geom.geom_type == "MultiPolygon":
            _polys = list(_geom.geoms)
        else:
            continue
        for _poly in _polys:
            _lons, _lats = _poly.exterior.coords.xy
            for _lon, _lat in zip(_lons, _lats):
                _x, _y = _proj.transform(_lon, _lat)
                coast_xs.append(_x)
                coast_ys.append(_y)
            coast_xs.append(None)
            coast_ys.append(None)

    with open("figures/fronts_data.js") as _f:
        _raw = _f.read()
    _geojson = json.loads(_raw.split("=", 1)[1].rstrip().rstrip(";"))
    fronts_proj = []
    for _feat in _geojson["features"]:
        _props = _feat["properties"]
        _fx, _fy = [], []
        for _line in _feat["geometry"]["coordinates"]:
            for _lon, _lat in _line:
                _x, _y = _proj.transform(_lon, _lat)
                _fx.append(_x)
                _fy.append(_y)
            _fx.append(None)
            _fy.append(None)
        fronts_proj.append({"name": _props["name"], "color": _props["color"], "x": _fx, "y": _fy})
    return coast_xs, coast_ys, fronts_proj, hex_boundaries_proj


@app.cell(hide_code=True)
def _(
    coast_xs,
    coast_ys,
    fronts_proj,
    go,
    hex_boundaries_proj,
    hex_r3_meta,
    selected_indices,
):
    spatial_fig = go.Figure()

    _bg_x, _bg_y = [], []
    for _cell, (_xs, _ys) in hex_boundaries_proj.items():
        _bg_x.extend(_xs + [None])
        _bg_y.extend(_ys + [None])

    spatial_fig.add_trace(
        go.Scatter(
            x=_bg_x, y=_bg_y, mode="lines",
            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
            hoverinfo="skip", showlegend=False,
        )
    )

    spatial_fig.add_trace(
        go.Scatter(
            x=coast_xs, y=coast_ys, mode="lines",
            line=dict(width=1, color="rgb(100,100,100)"),
            fill="toself", fillcolor="rgb(230,230,230)",
            hoverinfo="skip", showlegend=False,
        )
    )

    for _front in fronts_proj:
        spatial_fig.add_trace(
            go.Scatter(
                x=_front["x"], y=_front["y"], mode="lines",
                line=dict(width=1.5, color=_front["color"]),
                name=_front["name"], showlegend=True,
                hoverinfo="name",
            )
        )

    if selected_indices:
        _sel_meta = hex_r3_meta.iloc[sorted(selected_indices)]
        _sel_cells = _sel_meta["h3_cell"].unique()
        for _cell in _sel_cells:
            _xs, _ys = hex_boundaries_proj[_cell]
            _cell_rows = _sel_meta[_sel_meta["h3_cell"] == _cell]
            _years = sorted(_cell_rows["year"].unique())
            _seasons = sorted(_cell_rows["season"].unique())
            _months = sorted(_cell_rows["months"].dropna().unique())
            _hover = f"{_cell}<br>Years: {', '.join(str(y) for y in _years)}<br>Seasons: {', '.join(_seasons)}<br>Months: {', '.join(_months)}"
            spatial_fig.add_trace(
                go.Scatter(
                    x=_xs, y=_ys, mode="lines",
                    fill="toself", fillcolor="rgba(255, 65, 54, 0.5)",
                    line=dict(width=1.5, color="red"),
                    text=_hover, hoverinfo="text", showlegend=False,
                )
            )

    spatial_fig.update_layout(
        xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgb(210,230,250)",
        title="Spatial Location of Selected Hexbins" if selected_indices else "All H3 Hexbins (res 3)",
    )
    return (spatial_fig,)


@app.cell(hide_code=True)
def _(hex_r3_meta, mo, selected_indices):
    if selected_indices:
        _sel_meta = hex_r3_meta.iloc[sorted(selected_indices)][
            [
                "h3_cell",
                "year",
                "season",
                "months",
                "center_lat",
                "center_lon",
                "dist_to_antarctica_km",
                "mean_hadisst_sst",
                "mean_hadisst_ice",
            ]
        ].copy()
        mo.vstack(
            [
                mo.md(f"**Selected: {len(selected_indices)} hex-year samples from {_sel_meta['h3_cell'].nunique()} unique cells**"),
                mo.ui.dataframe(_sel_meta),
            ]
        )
    else:
        mo.md("*Select nodes in the Mapper graph to see metadata.*")
    return


@app.cell
def _(mapper_plot, mo, spatial_fig, umap_fig):
    mo.vstack([
        mo.hstack([mapper_plot, umap_fig], widths="equal"),
        spatial_fig,
    ])
    return


if __name__ == "__main__":
    app.run()
