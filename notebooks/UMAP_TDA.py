import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import umap
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.spatial.distance import pdist, squareform
    from sklearn.manifold import MDS


    return MDS, go, mo, np, pd, pdist, px, squareform, umap


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(pd):
    abund_df = pd.read_csv('data/abundance_processed.csv', index_col=0)
    abund_df = abund_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    meta_df = pd.read_csv('data/metadata/segment_metadata.csv', index_col=0)
    # Align to common segments
    _common = abund_df.index.intersection(meta_df.index)
    abund_df = abund_df.loc[_common]
    meta_df = meta_df.loc[_common]
    return abund_df, meta_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance matrix
    """)
    return


@app.cell
def _(abund_df, np, pdist, squareform):
    _X = abund_df.values
    bc_dist = squareform(pdist(_X, metric='braycurtis'))
    bc_dist = np.nan_to_num(bc_dist)  # guard against all-zero rows (0/0 → 0)
    return (bc_dist,)


@app.cell
def _():
    # put all x and y into one df, then plot them
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UMAP
    """)
    return


@app.cell
def _(abund_df):
    abund_df
    return


@app.cell
def _(abund_df, umap):
    _reducer_umap = umap.UMAP(metric='braycurtis', n_components=2, n_neighbors=15, random_state=42)
    umap_emb = _reducer_umap.fit_transform(abund_df)
    return (umap_emb,)


@app.cell
def _(meta_df, px, sns, umap_emb):
    df_umap = meta_df[['Year', 'Season', 'Latitude', 'Longitude', 'Water_Temperature']].copy()
    df_umap['UMAP1'] = umap_emb[:, 0]
    df_umap['UMAP2'] = umap_emb[:, 1]
    df_umap.index.name = 'Segment_ID'

    fig_umap = px.scatter(
        df_umap.reset_index(),
        x='UMAP1', y='UMAP2',
        color='Year',
        hover_name='Segment_ID',
        hover_data=['Season', 'Latitude', 'Longitude', 'Water_Temperature'],
        title='UMAP — Bray-Curtis distance',
        color_continuous_scale='Viridis',
        template='simple_white',
    )

    sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Year", alpha = 0.75)
    return (df_umap,)


@app.cell
def _(df_umap):
    df_umap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCoA
    """)
    return


@app.cell
def _(MDS, bc_dist, meta_df, px):
    _reducer_pcoa = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pcoa_emb = _reducer_pcoa.fit_transform(bc_dist)

    _df_pcoa = meta_df[['Year', 'Season', 'Latitude', 'Longitude', 'Water_Temperature']].copy()
    _df_pcoa['PCoA1'] = pcoa_emb[:, 0]
    _df_pcoa['PCoA2'] = pcoa_emb[:, 1]
    _df_pcoa.index.name = 'Segment_ID'

    _fig_pcoa = px.scatter(
        _df_pcoa.reset_index(),
        x='PCoA1', y='PCoA2',
        color='Year',
        hover_name='Segment_ID',
        hover_data=['Season', 'Latitude', 'Longitude', 'Water_Temperature'],
        title='PCoA (Classical MDS) — Bray-Curtis distance',
        color_continuous_scale='Viridis',
        template='simple_white',
    )
    _fig_pcoa.update_traces(marker=dict(size=4, opacity=0.7))
    _fig_pcoa
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### kMST
    """)
    return


@app.cell
def _():
    from multi_mst import KMST

    return (KMST,)


@app.cell
def _(KMST, abund_df, go, meta_df):
    _X_raw = abund_df.values
    _kmst_model = KMST(num_neighbors=3, epsilon=2.0).fit(_X_raw)
    _projector = _kmst_model.umap(repulsion_strength=1.0)

    _xs = _projector.embedding_[:, 0]
    _ys = _projector.embedding_[:, 1]
    _coo = _projector.graph_.tocoo()
    _src = _coo.row
    _tgt = _coo.col

    # Build edge traces (None separates pairs so Plotly doesn't connect them)
    _edge_x, _edge_y = [], []
    for _s, _t in zip(_src, _tgt):
        _edge_x += [float(_xs[_s]), float(_xs[_t]), None]
        _edge_y += [float(_ys[_s]), float(_ys[_t]), None]

    _years = meta_df['Year'].values

    _fig_kmst = go.Figure()
    _fig_kmst.add_trace(go.Scatter(
        x=_edge_x, y=_edge_y,
        mode='lines',
        line=dict(width=0.5, color='lightgrey'),
        hoverinfo='none',
        showlegend=False,
    ))
    _fig_kmst.add_trace(go.Scatter(
        x=list(_xs), y=list(_ys),
        mode='markers',
        marker=dict(
            size=4,
            color=_years,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Year'),
            opacity=0.8,
        ),
        text=list(meta_df.index),
        hovertemplate='%{text}<br>Year: %{marker.color}<extra></extra>',
        showlegend=False,
    ))
    _fig_kmst.update_layout(
        title='kMST — UMAP layout (raw abundance)',
        template='simple_white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    _fig_kmst
    return


if __name__ == "__main__":
    app.run()
