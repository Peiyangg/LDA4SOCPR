# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.7",
# ]
# ///

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # SO-CPR Data Exploration

    Explores the abundance data before LDA:
    1. Taxon-level sparsity and skewness
    2. Segment-level total abundance distribution
    3. Pinkerton log-normalisation preview
    4. LDA model selection metrics (Gensim K=2–30)
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Load Data
    """)
    return


@app.cell
def _(Path, pd):
    abundance = pd.read_csv(Path("data/abundance_processed.csv"), index_col=0)
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0)

    metadata = pd.read_csv(
        Path("data/metadata/segment_metadata.csv"), index_col=0
    )
    metadata["Segment_Length"] = pd.to_numeric(
        metadata["Segment_Length"], errors="coerce"
    )

    taxa_cols = abundance.columns.tolist()
    return abundance, metadata, taxa_cols


@app.cell
def _(abundance, mo):
    mo.md(f"""
    **Abundance table:** {abundance.shape[0]:,} segments × {abundance.shape[1]} taxa
    **Total counts:** {abundance.values.sum():,.0f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Taxon-level Statistics

    Two key properties of ecological count data:
    - **Skewness** — most taxa have right-skewed distributions (many zeros, few large values)
    - **Sparsity** — proportion of segments where each taxon is absent
    """)
    return


@app.cell
def _(abundance, mo, plt, taxa_cols):
    # Skewness per taxon
    skewness = abundance[taxa_cols].skew().sort_values(ascending=False)
    n_highly_skewed = (skewness > 1).sum()

    fig_skew, ax_skew = plt.subplots(figsize=(10, 3))
    ax_skew.bar(range(len(skewness)), skewness.values, color="steelblue", alpha=0.7, width=1.0)
    ax_skew.axhline(1, color="red", linestyle="--", linewidth=0.8, label="skewness = 1")
    ax_skew.set_xlabel("Taxon (sorted by skewness)")
    ax_skew.set_ylabel("Skewness")
    ax_skew.set_title(
        f"Skewness per taxon — {n_highly_skewed}/{len(taxa_cols)} highly skewed (>1)"
    )
    ax_skew.legend(fontsize=8)
    ax_skew.set_xlim(0, len(skewness))
    plt.tight_layout()

    mo.md(
        f"**{n_highly_skewed} / {len(taxa_cols)} taxa** are highly skewed (skewness > 1). "
        f"Mean skewness: {skewness.mean():.1f}, max: {skewness.max():.1f}"
    )
    return (skewness,)


@app.cell
def _(skewness):
    # Top 20 most skewed taxa
    skewness.head(20).to_frame("skewness")
    return


@app.cell
def _(abundance, plt, taxa_cols):
    # Zero proportion per taxon (split into 2 panels for readability)
    zero_prop = (abundance[taxa_cols] == 0).mean().sort_values(ascending=False)
    _n = len(zero_prop)
    _half = _n // 2

    fig_zero, (ax_z1, ax_z2) = plt.subplots(2, 1, figsize=(10, 6))

    for _ax, _data, _title in [
        (ax_z1, zero_prop.iloc[:_half], f"Taxa 1–{_half} (highest sparsity)"),
        (ax_z2, zero_prop.iloc[_half:], f"Taxa {_half+1}–{_n} (lower sparsity)"),
    ]:
        _ax.bar(range(len(_data)), _data.values, color="steelblue", alpha=0.7, width=1.0)
        _ax.axhline(0.95, color="red", linestyle="--", linewidth=0.8, label="95% zeros")
        _ax.set_ylabel("Zero proportion")
        _ax.set_title(_title)
        _ax.set_xlim(0, len(_data))
        _ax.set_ylim(0, 1.05)
        _ax.legend(fontsize=8)

    ax_z2.set_xlabel("Taxon (sorted by sparsity)")
    plt.suptitle("Proportion of segments with zero count per taxon", y=1.01)
    plt.tight_layout()
    fig_zero
    return (zero_prop,)


@app.cell
def _(mo, zero_prop):
    mo.md(f"""
    **Sparsity summary:**
    - Taxa with >95% zeros: {(zero_prop > 0.95).sum()}
    - Taxa with >80% zeros: {(zero_prop > 0.80).sum()}
    - Taxa with >50% zeros: {(zero_prop > 0.50).sum()}
    - Mean zero proportion: {zero_prop.mean():.1%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Total Abundance Distribution

    Comparing raw total abundance per segment vs. segment-length-corrected
    log concentration (Pinkerton normalisation).
    """)
    return


@app.cell
def _(abundance, metadata, np, plt, taxa_cols):
    _total_raw = abundance[taxa_cols].sum(axis=1)
    _total_normed = np.log(_total_raw / metadata["Segment_Length"] + 0.2)

    fig_hist, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(10, 4))

    ax_h1.hist(_total_raw.values, bins=100, color="steelblue", alpha=0.7)
    ax_h1.set_xlabel("Total counts per segment")
    ax_h1.set_ylabel("Number of segments")
    ax_h1.set_title("Raw total abundance")

    ax_h2.hist(_total_normed.values, bins=100, color="firebrick", alpha=0.7)
    ax_h2.set_xlabel("log(counts / segment length + 0.2)")
    ax_h2.set_ylabel("Number of segments")
    ax_h2.set_title("Log segment-corrected abundance")

    plt.suptitle("Total abundance per segment", y=1.02)
    plt.tight_layout()
    fig_hist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Pinkerton Log-Normalisation

    Per Pinkerton et al. (2020): for each taxon, log-transform the
    segment-length-corrected concentration where present, keeping zeros as-is.
    Adds a presence/absence indicator column for each taxon.
    """)
    return


@app.cell
def _(abundance, metadata, np, taxa_cols):
    abundance_normed = abundance[taxa_cols].copy().astype(float)

    for _col in taxa_cols:
        _mask = abundance[_col] > 0
        abundance_normed.loc[_mask, _col] = np.log(
            abundance.loc[_mask, _col] / metadata.loc[_mask, "Segment_Length"]
        )
        abundance_normed.loc[~_mask, _col] = 0
    return (abundance_normed,)


@app.cell
def _(abundance_normed):
    abundance_normed.describe().round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. LDA Model Selection (Gensim, K=2–30)

    Vertical lines mark the 4 candidate K values: **3, 7, 10, 16**.
    - **Perplexity** — lower = better fit (但可能 overfit)
    - **Coherence (c_v)** — higher = more interpretable topics
    """)
    return


@app.cell
def _(Path, pd):
    metrics = pd.read_csv(
        Path("LDA_gensim/lda_results/all_MC_metrics_16-30.csv"), index_col=0
    )
    metrics = metrics.sort_values("K").reset_index(drop=True)
    return (metrics,)


@app.cell
def _(metrics, plt):
    _candidates = [3, 7, 10, 16]
    _colors = ["firebrick", "darkorange", "forestgreen", "steelblue"]

    fig_metrics, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(12, 4))

    # Perplexity
    ax_p.plot(metrics["K"], metrics["Perplexity"], "o-", color="steelblue",
              markersize=4, linewidth=1.5)
    for _k, _col in zip(_candidates, _colors):
        ax_p.axvline(_k, color=_col, linestyle="--", linewidth=1, alpha=0.8, label=f"K={_k}")
    ax_p.set_xlabel("K (number of topics)")
    ax_p.set_ylabel("Perplexity")
    ax_p.set_title("Perplexity (lower = better fit)")
    ax_p.legend(fontsize=8)

    # Coherence
    ax_c.plot(metrics["K"], metrics["Coherence"], "o-", color="firebrick",
              markersize=4, linewidth=1.5)
    for _k, _col in zip(_candidates, _colors):
        ax_c.axvline(_k, color=_col, linestyle="--", linewidth=1, alpha=0.8, label=f"K={_k}")
    ax_c.set_xlabel("K (number of topics)")
    ax_c.set_ylabel("Coherence (c_v)")
    ax_c.set_title("Coherence (higher = more interpretable)")
    ax_c.legend(fontsize=8)

    plt.tight_layout()
    fig_metrics
    return


@app.cell
def _(metrics, mo):
    # Highlight candidate K values in a table
    _cands = metrics[metrics["K"].isin([3, 7, 10, 16])][["K", "Perplexity", "Coherence"]]
    mo.md("**Metrics at candidate K values:**")
    return


@app.cell
def _(metrics):
    metrics[metrics["K"].isin([3, 7, 10, 16])][
        ["K", "Perplexity", "Coherence"]
    ].set_index("K").round(4)
    return


if __name__ == "__main__":
    app.run()
