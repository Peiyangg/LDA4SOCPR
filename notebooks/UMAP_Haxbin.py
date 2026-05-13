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
    import seaborn as sns
    import matplotlib.pyplot as plt

    return mo, pd, plt, umap


@app.cell
def _(pd):
    hex_r2 = pd.read_csv('data/hex_features_res2.csv')
    hex_r3 = pd.read_csv('data/hex_features_res3.csv')

    hex_r2_meta = pd.read_csv('data/hex_metadata_res2.csv')
    hex_r3_meta = pd.read_csv('data/hex_metadata_res3.csv')

    hex_r2_effort = pd.read_csv('data/hex_effort_res2.csv', index_col=0)
    hex_r3_effort = pd.read_csv('data/hex_effort_res3.csv', index_col=0)
    return hex_r2, hex_r2_meta, hex_r3, hex_r3_effort, hex_r3_meta


@app.cell
def _(hex_r2, hex_r2_meta, hex_r3, hex_r3_meta):
    # For hex_r3_meta: create combined index, keep h3_cell as column
    hex_r3_meta.index = hex_r3_meta['h3_cell'].astype(str) + '_' + hex_r3_meta['year'].astype(str)
    # h3_cell column already exists, so it stays

    # For hex_r2_meta: create combined index, keep h3_cell as column  
    hex_r2_meta.index = hex_r2_meta['h3_cell'].astype(str) + '_' + hex_r2_meta['year'].astype(str)
    # h3_cell column already exists, so it stays

    # For hex_r3: create combined index, remove year column (pure abundance table)
    hex_r3.index = hex_r3['h3_cell'].astype(str) + '_' + hex_r3['year'].astype(str)
    hex_r3_clean = hex_r3.drop(columns=['h3_cell', 'year'])

    # For hex_r2: create combined index, remove year column (pure abundance table)
    hex_r2.index = hex_r2['h3_cell'].astype(str) + '_' + hex_r2['year'].astype(str)
    hex_r2_clean = hex_r2.drop(columns=['h3_cell', 'year'])
    return (hex_r3_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NMF
    """)
    return


@app.cell
def _():
    from cvanmf import denovo, combine, data, stability
    import plotnine as pn
    import pathlib

    return denovo, pathlib, pn


@app.cell
def _(denovo, hex_r3_clean):
    rank_res = denovo.rank_selection(
        x=hex_r3_clean,
        ranks=list(range(2, 11)),
        seed=4298,
        shuffles=100,
        progress_bar=True
    )
    return (rank_res,)


@app.cell
def _(rank_res):
    import pickle

    with open("nmf_results/rank_res.pkl", "wb") as f:
        pickle.dump(rank_res, f)
    return


@app.cell
def _(pathlib):
    PUB = True

    def figure_output(plot, name, pub):
        """Output a figure and the underlying data in PNG, PDF and TSV format."""
        if not pub:
            return plot
        prefix = pathlib.Path("nmf_results")
        out_path = prefix / name
        plot.save(out_path.with_suffix('.png'), dpi=300)
        plot.save(out_path.with_suffix('.pdf'))
        plot.data.to_csv(out_path.with_suffix('.tsv'), sep='\t')
        return plot

    return PUB, figure_output


@app.cell
def _(PUB, denovo, figure_output, pn, rank_res):
    plt_ranksel = (
        denovo.plot_rank_selection(rank_res, jitter=False) 
        + pn.guides(fill = "none")
        + pn.theme(figure_size=(2.66 * 2, 1.8))
        + pn.ggtitle("Bicrossvalidation Rank Selection")
        + pn.xlab("Rank")
        + pn.ylab("")
        # We're adding a little more room for the stars - the internal calculation doesn't always get the right space for custom size plots
        + pn.scale_y_continuous(expand=[0.1, 0, 0.1, 0])
    )

    figure_output(plt_ranksel, "a_ranksel", PUB)
    # You can show the figure in the notebook by using it as the final line of the cell
    plt_ranksel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UMAP_r3
    """)
    return


@app.cell
def _(hex_r3_clean, umap):
    _reducer_umap = umap.UMAP(metric='braycurtis', n_components=2, n_neighbors=5, random_state=42)
    umap_emb = _reducer_umap.fit_transform(hex_r3_clean)
    return (umap_emb,)


@app.cell
def _(hex_r3_meta, umap_emb):
    df_umap = hex_r3_meta[['year', 'season', 'center_lat', 'center_lon', 'mean_water_temperature','mean_hadisst_ice', 'mean_hadisst_sst']].copy()
    df_umap['UMAP1'] = umap_emb[:, 0]
    df_umap['UMAP2'] = umap_emb[:, 1]
    df_umap.index.name = 'Segment_ID'
    return (df_umap,)


@app.cell
def _(df_umap, mo):
    mo.ui.data_explorer(
        df_umap,
        x="UMAP1",
        y="UMAP2",
        color="center_lat",
    )
    return


@app.cell
def _():
    # sample time more than 4
    return


@app.cell
def _(hex_r3_effort, plt):
    # Count non-zero values per row
    non_zero_counts = (hex_r3_effort != 0).sum(axis=1)
    rows_with_multiple_values = hex_r3_effort[non_zero_counts > 1]
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_counts, bins=range(0, non_zero_counts.max() + 2), edgecolor='black')
    plt.xlabel('Number of columns with values (non-zero)')
    plt.ylabel('Frequency')
    plt.title('Distribution of non-zero columns per row')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-Year UMAP Analysis
    """)
    return


@app.cell
def _(hex_r3_clean, hex_r3_meta):
    # Get unique years actually present in the data
    _unique_years = sorted(hex_r3_meta['year'].unique())
    _min_year, _max_year = min(_unique_years), max(_unique_years)

    # Build 5-year bins working BACKWARDS from max_year so the latest bin
    # always contains a full 5 years of actual data.
    _bin_edges = []
    _end = _max_year
    while _end >= _min_year:
        _start = _end - 4  # 5-year window (inclusive both ends)
        _bin_edges.append((_start, _end))
        _end = _start - 1
    _bin_edges = sorted(_bin_edges)  # chronological order

    # Create subdataframes for each 5-year period, keeping only bins that
    # actually contain samples.
    subdf_dict = {}
    for _start_year, _end_year in _bin_edges:
        _mask = (hex_r3_meta['year'] >= _start_year) & (hex_r3_meta['year'] <= _end_year)
        if _mask.sum() == 0:
            continue
        _period_name = f"{_start_year}-{_end_year}"
        subdf_dict[_period_name] = {
            'data': hex_r3_clean[_mask],
            'meta': hex_r3_meta[_mask],
            'years': f"{_start_year}-{_end_year}",
            'actual_years': sorted(hex_r3_meta.loc[_mask, 'year'].unique().tolist()),
        }

    year_bins = _bin_edges

    print(f"Created {len(subdf_dict)} 5-year periods (anchored on max year {_max_year}):")
    for _period, _info in subdf_dict.items():
        print(f"  {_period}: {len(_info['data']):>4d} samples | actual years present: {_info['actual_years']}")
    return (subdf_dict,)


@app.cell
def _(subdf_dict, umap):
    # Get the latest period (last key in dictionary)
    _periods = list(subdf_dict.keys())
    latest_period = _periods[-1]

    print(f"Training UMAP on latest period: {latest_period}")

    # Fit UMAP on the latest 5 years
    reducer_5yr = umap.UMAP(
        metric='braycurtis', 
        n_components=2, 
        n_neighbors=15, 
        min_dist=0.1,
        random_state=42
    )

    latest_data = subdf_dict[latest_period]['data']
    latest_embedding = reducer_5yr.fit_transform(latest_data)

    print(f"Latest period shape: {latest_data.shape}")
    print(f"Embedding shape: {latest_embedding.shape}")
    return latest_period, reducer_5yr


@app.cell
def _(latest_period, pd, reducer_5yr, subdf_dict):
    # Transform all other periods using the fitted UMAP
    all_embeddings = {}

    for _period_name, _period_info in subdf_dict.items():
        if _period_name == latest_period:
            # Use the already computed embedding for latest period
            _embedding = reducer_5yr.transform(_period_info['data'])
        else:
            # Transform other periods
            print(f"Transforming period: {_period_name}")
            _embedding = reducer_5yr.transform(_period_info['data'])

        # Store embedding with metadata
        all_embeddings[_period_name] = {
            'embedding': _embedding,
            'meta': _period_info['meta'],
            'n_samples': len(_embedding)
        }

    # Combine all embeddings into a single dataframe
    _embedding_list = []
    for _period_name, _emb_info in all_embeddings.items():
        _temp_df = _emb_info['meta'][['year', 'season', 'center_lat', 'center_lon']].copy()
        _temp_df['UMAP1'] = _emb_info['embedding'][:, 0]
        _temp_df['UMAP2'] = _emb_info['embedding'][:, 1]
        _temp_df['period'] = _period_name
        _embedding_list.append(_temp_df)

    df_umap_5yr = pd.concat(_embedding_list, axis=0)

    print(f"\nTotal samples in combined embedding: {len(df_umap_5yr)}")
    return all_embeddings, df_umap_5yr


@app.cell
def _(df_umap_5yr, mo):
    # Interactive explorer for 5-year UMAP results
    mo.ui.data_explorer(
        df_umap_5yr,
        x="UMAP1",
        y="UMAP2",
        color="period",
    )
    return


@app.cell(hide_code=True)
def _(df_umap_5yr, pd, plt):
    # Plot UMAP colored by period
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by period
    _axes[0].scatter(
        df_umap_5yr['UMAP1'], 
        df_umap_5yr['UMAP2'],
        c=pd.Categorical(df_umap_5yr['period']).codes,
        cmap='tab10',
        alpha=0.6,
        s=20
    )
    _axes[0].set_xlabel('UMAP1')
    _axes[0].set_ylabel('UMAP2')
    _axes[0].set_title('UMAP Embedding - Colored by 5-Year Period')

    # Add legend for periods
    _periods = sorted(df_umap_5yr['period'].unique())
    _handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=plt.cm.tab10(_j / len(_periods)), 
                         markersize=8, label=_p)
              for _j, _p in enumerate(_periods)]
    _axes[0].legend(handles=_handles, title='Period', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Colored by year
    _scatter2 = _axes[1].scatter(
        df_umap_5yr['UMAP1'], 
        df_umap_5yr['UMAP2'],
        c=df_umap_5yr['year'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    _axes[1].set_xlabel('UMAP1')
    _axes[1].set_ylabel('UMAP2')
    _axes[1].set_title('UMAP Embedding - Colored by Year')
    plt.colorbar(_scatter2, ax=_axes[1], label='Year')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(all_embeddings, df_umap_5yr, plt):
    # Small multiples: one subplot per 5-year period
    _periods = sorted(all_embeddings.keys())
    _n_periods = len(_periods)

    # Grid layout: prefer 3 columns
    _ncols = 3
    _nrows = (_n_periods + _ncols - 1) // _ncols

    # Global axis limits (shared across all subplots)
    _x_min, _x_max = df_umap_5yr['UMAP1'].min(), df_umap_5yr['UMAP1'].max()
    _y_min, _y_max = df_umap_5yr['UMAP2'].min(), df_umap_5yr['UMAP2'].max()
    _pad_x = (_x_max - _x_min) * 0.05
    _pad_y = (_y_max - _y_min) * 0.05

    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 4 * _nrows),
                                sharex=True, sharey=True)
    _axes_flat = _axes.flatten() if _n_periods > 1 else [_axes]

    for _k, _period in enumerate(_periods):
        _ax = _axes_flat[_k]

        # Background: all other periods in light grey
        _ax.scatter(
            df_umap_5yr['UMAP1'],
            df_umap_5yr['UMAP2'],
            c='lightgrey',
            s=8,
            alpha=0.3,
            rasterized=True,
        )

        # Foreground: current period colored
        _emb = all_embeddings[_period]['embedding']
        _meta = all_embeddings[_period]['meta']
        _sc = _ax.scatter(
            _emb[:, 0],
            _emb[:, 1],
            c=_meta['year'],
            cmap='viridis',
            s=18,
            alpha=0.8,
            edgecolor='k',
            linewidth=0.2,
        )

        _ax.set_title(f"{_period} (n={len(_emb)})")
        _ax.set_xlim(_x_min - _pad_x, _x_max + _pad_x)
        _ax.set_ylim(_y_min - _pad_y, _y_max + _pad_y)
        _ax.set_xlabel('UMAP1')
        _ax.set_ylabel('UMAP2')
        plt.colorbar(_sc, ax=_ax, label='Year', shrink=0.8)

    # Hide unused subplots
    for _j in range(_n_periods, len(_axes_flat)):
        _axes_flat[_j].axis('off')

    _fig.suptitle('UMAP Embedding by 5-Year Period\n(fitted on latest period, transformed onto others)',
                   fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(all_embeddings, df_umap_5yr, plt):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Small multiples: one subplot per 5-year period
    _periods = sorted(all_embeddings.keys())
    _n_periods = len(_periods)

    # Grid layout: prefer 3 columns
    _ncols = 3
    _nrows = (_n_periods + _ncols - 1) // _ncols

    # SHARED axis limits
    _x_min, _x_max = df_umap_5yr['UMAP1'].min(), df_umap_5yr['UMAP1'].max()
    _y_min, _y_max = df_umap_5yr['UMAP2'].min(), df_umap_5yr['UMAP2'].max()
    _pad_x = (_x_max - _x_min) * 0.05
    _pad_y = (_y_max - _y_min) * 0.05

    # FIXED color scale: 1991-2019 for all subplots
    _norm = Normalize(vmin=1991, vmax=2019)
    _cmap = plt.cm.viridis

    _fig, _axes = plt.subplots(
        _nrows, _ncols,
        figsize=(5 * _ncols, 4 * _nrows),
        sharex=True, sharey=True,
    )
    _axes_flat = _axes.flatten() if _n_periods > 1 else [_axes]

    for _k, _period in enumerate(_periods):
        _ax = _axes_flat[_k]

        # Background: all other periods in light grey
        _ax.scatter(
            df_umap_5yr['UMAP1'],
            df_umap_5yr['UMAP2'],
            c='lightgrey',
            s=8,
            alpha=0.3,
            rasterized=True,
        )

        # Foreground: current period, using FIXED 1991-2019 norm
        _emb = all_embeddings[_period]['embedding']
        _meta = all_embeddings[_period]['meta']
        _ax.scatter(
            _emb[:, 0],
            _emb[:, 1],
            c=_meta['year'],
            cmap=_cmap,
            norm=_norm,
            s=18,
            alpha=0.85,
            edgecolor='k',
            linewidth=0.2,
        )

        _ax.set_title(f"{_period} (n={len(_emb)})")
        _ax.set_xlim(_x_min - _pad_x, _x_max + _pad_x)
        _ax.set_ylim(_y_min - _pad_y, _y_max + _pad_y)
        _ax.set_xlabel('UMAP1')
        _ax.set_ylabel('UMAP2')

    # Hide any unused subplots
    for _j in range(_n_periods, len(_axes_flat)):
        _axes_flat[_j].axis('off')

    # ONE shared colorbar for all subplots (1991-2019 range)
    _sm = ScalarMappable(norm=_norm, cmap=_cmap)
    _sm.set_array([])
    _cbar = _fig.colorbar(
        _sm,
        ax=_axes_flat,
        orientation='vertical',
        fraction=0.02,
        pad=0.02,
        label='Year',
    )

    _fig.suptitle(
        'UMAP Embedding by 5-Year Period\n(fitted on latest period, transformed onto others)',
        fontsize=14, y=1.00,
    )
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(all_embeddings, df_umap_5yr, plt):
    # Small multiples: one subplot per 5-year period
    # Each subplot has its OWN color scale (based on years in that period)
    _periods = sorted(all_embeddings.keys())
    _n_periods = len(_periods)

    # Grid layout: prefer 3 columns
    _ncols = 3
    _nrows = (_n_periods + _ncols - 1) // _ncols

    # Global axis limits (shared across all subplots)
    _x_min, _x_max = df_umap_5yr['UMAP1'].min(), df_umap_5yr['UMAP1'].max()
    _y_min, _y_max = df_umap_5yr['UMAP2'].min(), df_umap_5yr['UMAP2'].max()
    _pad_x = (_x_max - _x_min) * 0.05
    _pad_y = (_y_max - _y_min) * 0.05

    _fig, _axes = plt.subplots(
        _nrows, _ncols,
        figsize=(5 * _ncols, 4 * _nrows),
        sharex=True, sharey=True,
    )
    _axes_flat = _axes.flatten() if _n_periods > 1 else [_axes]

    for _k, _period in enumerate(_periods):
        _ax = _axes_flat[_k]

        # Background: all other periods in light grey
        _ax.scatter(
            df_umap_5yr['UMAP1'],
            df_umap_5yr['UMAP2'],
            c='lightgrey',
            s=8,
            alpha=0.3,
            rasterized=True,
        )

        # Foreground: current period colored by its OWN year range
        _emb = all_embeddings[_period]['embedding']
        _meta = all_embeddings[_period]['meta']
        _sc = _ax.scatter(
            _emb[:, 0],
            _emb[:, 1],
            c=_meta['year'],
            cmap='viridis',
            s=18,
            alpha=0.85,
            edgecolor='k',
            linewidth=0.2,
        )

        _ax.set_title(f"{_period} (n={len(_emb)})")
        _ax.set_xlim(_x_min - _pad_x, _x_max + _pad_x)
        _ax.set_ylim(_y_min - _pad_y, _y_max + _pad_y)
        _ax.set_xlabel('UMAP1')
        _ax.set_ylabel('UMAP2')
        plt.colorbar(_sc, ax=_ax, label='Year', shrink=0.8)

    # Hide unused subplots
    for _j in range(_n_periods, len(_axes_flat)):
        _axes_flat[_j].axis('off')

    _fig.suptitle(
        'UMAP by 5-Year Period — Per-Subplot Color Scale',
        fontsize=14, y=1.00,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
