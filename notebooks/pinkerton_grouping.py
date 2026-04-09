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
    # Pinkerton 2020 Taxonomic Grouping

    Aggregates the SO-CPR abundance table into the 11 ecological groups
    defined in **Pinkerton et al. (2020)** (Table 1).

    | Group | Name |
    |-------|------|
    | 1 | *Oithona similis* (+ unidentified Cyclopoida) |
    | 2 | Copepoda (all other copepods) |
    | 3 | Amphipoda |
    | 4 | Chaetognatha |
    | 5 | Euphausiidae (krill, all life stages) |
    | 6 | Foraminifera |
    | 7 | *Fritillaria* spp. |
    | 8 | *Oikopleura* spp. |
    | 9 | Ostracoda |
    | 10 | Pteropods |
    | 11 | Other |
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Load Data
    """)
    return


@app.cell
def _(Path, pd):
    # Load abundance table (segments × taxa, integer counts)
    abundance = pd.read_csv(
        Path("data/abundance_processed.csv"), index_col=0
    )
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Load taxonomy annotations for programmatic group assignment
    taxa = pd.read_csv(Path("data/taxon_annotations.csv"))
    taxa = taxa.set_index("ID")

    all_taxa_cols = abundance.columns.tolist()
    return abundance, all_taxa_cols, taxa


@app.cell
def _(abundance, mo):
    mo.md(f"""
    **Loaded:** {abundance.shape[0]:,} segments × {abundance.shape[1]} taxa columns
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Define Pinkerton Groups
    """)
    return


@app.cell
def _(all_taxa_cols, taxa):
    # ── Group 1: Oithona similis + unidentified Cyclopoida ──────────────────
    # Per paper: "dominated by O. similis (97%), incl. unidentified Cyclopoida"
    G1 = [c for c in all_taxa_cols if "Oithona" in c or "Cyclopoida" in c]

    # ── Group 3: Amphipoda ───────────────────────────────────────────────────
    _amp_ids = taxa[taxa["order"] == "Amphipoda"].index.tolist()
    G3 = [c for c in all_taxa_cols if c in _amp_ids]

    # ── Group 4: Chaetognatha ────────────────────────────────────────────────
    # Includes Sagittidae, Eukrohniidae — strictly phylum Chaetognatha
    G4 = [
        c for c in all_taxa_cols
        if any(k in c for k in [
            "Chaetognatha", "Sagittidae", "Eukrohnia", "Pseudosagitta",
            "Solidosagitta", "Sagitta"
        ])
    ]

    # ── Group 5: Euphausiidae (all krill, all life stages) ───────────────────
    _euph_ids = taxa[taxa["family"] == "Euphausiidae"].index.tolist()
    G5 = [c for c in all_taxa_cols if c in _euph_ids]

    # ── Group 6: Foraminifera ────────────────────────────────────────────────
    _foram_ids = taxa[taxa["order"] == "Foraminifera"].index.tolist()
    G6_by_taxa = [c for c in all_taxa_cols if c in _foram_ids]
    # Also catch by name (some may not resolve in WoRMS)
    G6_by_name = [
        c for c in all_taxa_cols
        if any(k in c for k in [
            "Foraminifera", "Globigerina", "Globigerinita",
            "Neogloboquadrina", "Turborotalita", "Globorotalia",
            "Radiozoa"
        ])
    ]
    G6 = list(set(G6_by_taxa + G6_by_name))

    # ── Group 7: Fritillaria spp. ────────────────────────────────────────────
    G7 = [c for c in all_taxa_cols if "Fritillaria" in c]

    # ── Group 8: Oikopleura spp. ─────────────────────────────────────────────
    G8 = [c for c in all_taxa_cols if "Oikopleura" in c]

    # ── Group 9: Ostracoda ───────────────────────────────────────────────────
    G9 = [c for c in all_taxa_cols if "Ostracoda" in c]

    # ── Group 10: Pteropods ──────────────────────────────────────────────────
    # Pelagic Gastropoda: Limacina, Clio, Clione, Spongiobranchaea + indet
    _ptero_ids = taxa[taxa["order"] == "Pteropoda"].index.tolist()
    G10_by_taxa = [c for c in all_taxa_cols if c in _ptero_ids]
    G10_by_name = [
        c for c in all_taxa_cols
        if any(k in c for k in [
            "Pteropoda", "Limacina", "Clio ", "Clio p", "Clio s",
            "Clione", "Spongiobranchaea"
        ])
    ]
    G10 = list(set(G10_by_taxa + G10_by_name))

    # ── Group 2: All other Copepoda ──────────────────────────────────────────
    _cop_ids = taxa[taxa["class"] == "Copepoda"].index.tolist()
    _cop_all = [c for c in all_taxa_cols if c in _cop_ids]
    G2 = [c for c in _cop_all if c not in G1]

    # ── Group 11: Other ──────────────────────────────────────────────────────
    # Everything not assigned to groups 1-10
    _assigned = set(G1 + G2 + G3 + G4 + G5 + G6 + G7 + G8 + G9 + G10)
    G11 = [c for c in all_taxa_cols if c not in _assigned]

    # Build mapping dict: group_name → list of columns
    GROUP_MAPPING = {
        "Oithona similis":  G1,
        "Copepoda":         G2,
        "Amphipoda":        G3,
        "Chaetognatha":     G4,
        "Euphausiidae":     G5,
        "Foraminifera":     G6,
        "Fritillaria spp.": G7,
        "Oikopleura spp.":  G8,
        "Ostracoda":        G9,
        "Pteropods":        G10,
        "Other":            G11,
    }
    return (GROUP_MAPPING,)


@app.cell
def _(GROUP_MAPPING, mo, pd):
    # Summary table of group assignments
    _summary = pd.DataFrame([
        {"Group": f"{_i+1}. {_name}", "# taxa columns": len(_cols)}
        for _i, (_name, _cols) in enumerate(GROUP_MAPPING.items())
    ])
    mo.md(
        f"**Total assigned:** "
        f"{sum(len(v) for v in GROUP_MAPPING.values())} columns across 11 groups\n\n"
    )
    return


@app.cell
def _(GROUP_MAPPING, pd):
    # Detailed assignment table
    pd.DataFrame([
        {"Group": f"{_i+1}. {_name}", "# taxa columns": len(_cols)}
        for _i, (_name, _cols) in enumerate(GROUP_MAPPING.items())
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Build Grouped Abundance Table
    """)
    return


@app.cell
def _(GROUP_MAPPING, abundance, pd):
    # Sum abundance across columns within each group
    grouped_rows = {}
    for _group_name, _cols in GROUP_MAPPING.items():
        _valid_cols = [c for c in _cols if c in abundance.columns]
        if _valid_cols:
            grouped_rows[_group_name] = abundance[_valid_cols].sum(axis=1)
        else:
            grouped_rows[_group_name] = 0

    abundance_pinkerton = pd.DataFrame(grouped_rows, index=abundance.index)

    abundance_pinkerton
    return (abundance_pinkerton,)


@app.cell
def _(abundance, abundance_pinkerton, mo):
    # Sanity check: total counts should be preserved
    _orig_total = abundance.values.sum()
    _new_total = abundance_pinkerton.values.sum()
    _diff = abs(_orig_total - _new_total)

    mo.md(f"""
    **Sanity check:**
    - Original total counts: {_orig_total:,.0f}
    - Grouped total counts:  {_new_total:,.0f}
    - Difference: {_diff:.0f} {'✅ (no loss)' if _diff == 0 else '⚠️ check assignment'}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Group Composition Overview
    """)
    return


@app.cell
def _(abundance_pinkerton, plt):
    # Total abundance per group across all segments
    _totals = abundance_pinkerton.sum().sort_values(ascending=True)

    fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
    _bars = ax_comp.barh(_totals.index, _totals.values, color="steelblue", alpha=0.8)
    ax_comp.set_xlabel("Total abundance (all segments)")
    ax_comp.set_title("Total abundance by Pinkerton 2020 group")
    ax_comp.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )
    plt.tight_layout()
    fig_comp
    return


@app.cell
def _(abundance_pinkerton, plt):
    # Relative composition pie chart
    _totals = abundance_pinkerton.sum()
    _pct = _totals / _totals.sum() * 100
    _pct_sorted = _pct.sort_values(ascending=False)

    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    _wedges, _texts, _autotexts = ax_pie.pie(
        _pct_sorted.values,
        labels=_pct_sorted.index,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=90,
        pctdistance=0.75,
    )
    for _t in _autotexts:
        _t.set_fontsize(8)
    ax_pie.set_title("Relative composition across all segments\n(Pinkerton 2020 groups)")
    plt.tight_layout()
    fig_pie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Export
    """)
    return


@app.cell
def _(abundance_pinkerton):
    abundance_pinkerton
    return


@app.cell
def _(Path, abundance_pinkerton, mo, plt):
    _output_dir = Path("data")
    _fig_dir = Path("figures")
    _fig_dir.mkdir(exist_ok=True)

    # Save grouped abundance table (Segment_ID as index, 11 groups as columns)
    abundance_pinkerton.to_csv(_output_dir / "abundance_pinkerton2020.csv")

    # Save figures at 300 dpi, 8×6 inches (matching R script)
    # Composition bar chart
    _totals = abundance_pinkerton.sum().sort_values(ascending=True)
    _fig_bar, _ax_bar = plt.subplots(figsize=(8, 5))
    _ax_bar.barh(_totals.index, _totals.values, color="steelblue", alpha=0.8)
    _ax_bar.set_xlabel("Total abundance (all segments)")
    _ax_bar.set_title("Total abundance by Pinkerton 2020 group")
    _ax_bar.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )
    plt.tight_layout()
    _fig_bar.savefig(_fig_dir / "pinkerton_groups_composition.png", dpi=300)
    plt.close(_fig_bar)

    mo.md(f"""
    **Saved outputs:**
    - `data/abundance_pinkerton2020.csv`
      ({abundance_pinkerton.shape[0]:,} segments × {abundance_pinkerton.shape[1]} groups)
    - `figures/pinkerton_groups_composition.png` (300 dpi, 8×5 in)
    """)
    return


if __name__ == "__main__":
    app.run()
