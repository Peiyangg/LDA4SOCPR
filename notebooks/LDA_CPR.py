import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(pd):
    abundance = pd.read_csv('data/abundance_processed.csv', index_col=0)
    taxa = pd.read_csv('data/taxon_annotations.csv')
    metadata = pd.read_csv('data/metadata/segment_metadata.csv', index_col=0)
    return abundance, metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LDA
    """)
    return


@app.cell
def _():
    import lda4microbiome

    return


@app.cell
def _():
    mallet_path = 'Mallet/bin/mallet'
    return


@app.cell
def _(abundance):
    from gensim.corpora import Dictionary

    # Convert to integer counts (LDA needs discrete counts)
    # _abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Build processed_texts: each segment = a "document" where taxa are repeated by count
    processed_texts = []
    for _idx in range(len(abundance)):
        _row = abundance.iloc[_idx]
        _doc = []
        for _taxon, _count in _row.items():
            if _count > 0:
                _doc.extend([_taxon] * _count)
        processed_texts.append(_doc)

    # Create Gensim dictionary and corpus
    dictionary = Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    print(f"Documents (segments): {len(corpus)}")
    print(f"Vocabulary (taxa): {len(dictionary)}")
    print(f"Total counts: {abundance.values.sum():,}")
    return corpus, dictionary, processed_texts


@app.cell
def _(abundance, corpus, dictionary, processed_texts):
    from lda4microbiome import LDATrainer

    trainer = LDATrainer(
        base_directory="LDA_gensim/",
        implementation="gensim",
        passes=20,
        iterations=400,
        alpha="auto",
        eta="auto",
        random_state=42,
    )

    # Load custom data directly (skip TaxonomyProcessor)
    trainer.set_custom_gensim_data(
        dictionary=dictionary,
        corpus=corpus,
        processed_texts=processed_texts,
        sample_index=abundance.index,
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train_models(MC_range=list(range(16, 31)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### metrices
    """)
    return


@app.cell
def _():
    import plotly.express as px

    return (px,)


@app.cell
def _(pd):
    metrices = pd.read_csv('/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/NatureScience/POLARY/CPR_DataAnalysis/LDA_gensim/lda_results/all_MC_metrics_16-30.csv', index_col=0)
    return (metrices,)


@app.cell
def _(pd):
    abundance_11 = pd.read_csv('data/abundance_pinkerton2020.csv', index_col=0)
    return


@app.cell
def _(abundance, np):
    # skewness for each taxa column
    taxa_cols = abundance.select_dtypes(include=np.number).columns

    skewness = abundance[taxa_cols].skew().sort_values(ascending=False)
    print(skewness)

    # anything above 1 is considered highly skewed
    highly_skewed = skewness[skewness > 1]
    print(f"\n{len(highly_skewed)} highly skewed taxa out of {len(taxa_cols)}")
    return (taxa_cols,)


@app.cell
def _(abundance, plt):
    _taxa_cols = list(abundance)[250:]

    # proportion of zeros per taxa
    zero_prop = (abundance[_taxa_cols] == 0).mean().sort_values(ascending=False)
    print(zero_prop)

    # visualise
    zero_prop.plot(kind='bar', figsize=(15, 4))
    plt.title('Proportion of zeros per taxon')
    plt.ylabel('Proportion of segments with zero count')
    plt.axhline(0.95, color='red', linestyle='--', label='95% zeros')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(abundance, go, make_subplots, metadata, np, taxa_cols):
    _fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Raw Total Abundance',
                                        'Log Normalised Total Abundance'])

    # raw
    total_raw = abundance[taxa_cols].sum(axis=1)

    # correct: sum first THEN normalise
    total_normed = np.log(total_raw / metadata['Segment_Length'] + 0.2)

    _fig.add_trace(
        go.Histogram(x=total_raw, nbinsx=100, name='raw'),
        row=1, col=1
    )

    _fig.add_trace(
        go.Histogram(x=total_normed, nbinsx=100, name='normed'),
        row=1, col=2
    )

    _fig.update_layout(
        title='Total Abundance per Segment',
        template='plotly_white',
        showlegend=False
    )

    _fig.show()
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _(abundance, metadata, np, taxa_cols):
    # apply Pinkerton normalisation to each taxa column independently
    abundance_normed = abundance.copy()
    abundance_normed[taxa_cols] = abundance_normed[taxa_cols].astype(float)

    for col in taxa_cols:
        # presence absence
        abundance_normed[f'{col}_pa'] = (abundance[col] > 0).astype(int)
    
        # log concentration only where present
        mask = abundance[col] > 0
        abundance_normed.loc[mask, col] = np.log(
            abundance.loc[mask, col] / metadata.loc[mask, 'Segment_Length']
        )
        abundance_normed.loc[~mask, col] = 0
    return (abundance_normed,)


@app.cell
def _(abundance_normed):
    abundance_normed
    return


@app.cell
def _(metrices, px):
    fig = px.line(metrices, x="K", y="Perplexity", title='Perplexity of K')
    fig.show()
    return


@app.cell
def _(metrices, px):
    _fig = px.line(metrices, x="K", y="Coherence", title='Coherence of K')
    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Export Topic–Segment Probability Tables

    For K = 3, 7, 10, 16: load the saved sample probability matrix,
    transpose so **rows = segments, columns = topics**, and save as CSV.
    """)
    return


@app.cell
def _(pd):
    from pathlib import Path

    _output_dir = Path("LDA_gensim/lda_results/topic_distributions")
    _output_dir.mkdir(exist_ok=True)

    _candidate_ks = [3, 7, 10, 16]
    _saved = []

    for _k in _candidate_ks:
        _src = Path(f"LDA_gensim/lda_results/MC_Sample/MC_Sample_probabilities{_k}.csv")
        # File is saved as topics × segments — transpose to segments × topics
        _df = pd.read_csv(_src, index_col=0).T
        _df.index.name = "Segment_ID"
        _out = _output_dir / f"K{_k}_segment_topic_probs.csv"
        _df.to_csv(_out)
        _saved.append((_k, _df.shape, str(_out)))
        print(f"K={_k}: saved {_df.shape[0]:,} segments × {_df.shape[1]} topics → {_out.name}")

    return


if __name__ == "__main__":
    app.run()
