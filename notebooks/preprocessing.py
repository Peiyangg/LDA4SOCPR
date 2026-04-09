# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "matplotlib>=3.7",
#     "pyworms>=0.1",
#     "scipy>=1.11",
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
    # SO-CPR Data Preprocessing Pipeline

    This notebook preprocesses the Southern Ocean Continuous Plankton Recorder
    (SO-CPR) dataset. It translates the original R pipeline into Python, covering:

    1. **Load & inspect** the raw CPR data
    2. **Parse taxon names** from species columns
    3. **Annotate taxonomy** via WoRMS (World Register of Marine Species)
    4. **Calculate indices** (total abundance, Salp/Euphausiid balance)
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from pathlib import Path

    return Path, np, pd, plt, re


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Load & Inspect Data
    """)
    return


@app.cell(hide_code=True)
def _(Path, pd):
    # Load the raw CPR CSV
    data_path = Path("data/AADC-00099_29August2025.csv")
    cpr_raw = pd.read_csv(data_path, low_memory=False)

    # Define metadata columns (leading + trailing)
    _leading_meta = [
        "Tow_Number",
        "Ship_Code",
        "Time",
        "Date",
        "Month",
        "Year",
        "Season",
        "Latitude",
        "Longitude",
        "Segment_No.",
        "Segment_Length",
    ]
    _trailing_meta = [
        "Total abundance",
        "Phytoplankton_Colour_Index",
        "Fluorescence",
        "Salinity",
        "Water_Temperature",
        "Photosynthetically_Active_Radiation",
    ]
    colnames_metadata = _leading_meta + _trailing_meta

    # Species columns = everything not in metadata
    colnames_species = [c for c in cpr_raw.columns if c not in colnames_metadata]

    # Create a unique segment identifier for use as index
    cpr_raw["Segment_ID"] = (
        cpr_raw["Ship_Code"].astype(str)
        + "_" + cpr_raw["Tow_Number"].astype(str)
        + "_" + cpr_raw["Date"].astype(str)
        + "_" + cpr_raw["Time"].astype(str)
    )

    return colnames_metadata, colnames_species, cpr_raw


@app.cell
def _(colnames_species, cpr_raw, mo):
    mo.md(f"""
    **Dataset overview:**
    - **Rows (segments):** {len(cpr_raw):,}
    - **Total columns:** {len(cpr_raw.columns)}
    - **Metadata columns:** {len(cpr_raw.columns) - len(colnames_species)}
    - **Species/taxon columns:** {len(colnames_species)}
    - **Year range:** {cpr_raw['Year'].min()} – {cpr_raw['Year'].max()}
    """)
    return


@app.cell
def _(cpr_raw):
    cpr_raw.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Parse Taxon Names

    Each species column name encodes genus, species, life stage, and/or qualifier
    information. We parse these to prepare for WoRMS taxonomy lookup.
    """)
    return


@app.cell(hide_code=True)
def _(re):
    # Life stages recognized in CPR data
    LIFE_STAGES = {
        "egg",
        "nauplius",
        "metanauplius",
        "zoea",
        "megalopa",
        "phyllosoma",
        "juv",
        "natant",
        "larvae",
        "small",
        "calyptopis",
        "furcilia",
        "cyprid",
        "nectophore",
    }

    def parse_taxon_name(name: str) -> dict:
        """Parse a CPR species column name into taxonomic components.

        Handles space-separated names (new CSV format).
        Returns dict with keys: ID, Genus, Species, LifeStage, Qualifier,
        HigherTaxon, RawID as applicable.
        """
        result = {"ID": name}
        parts = name.split(" ")

        # Egg cases: "Egg indet", "Egg mass"
        if name.startswith("Egg"):
            result["LifeStage"] = "egg"
            return result

        # Life stage codes like C1, F2: e.g. "Euphausia superba F3"
        if re.search(r" [CF]\d$", name) and len(parts) >= 3:
            result["Genus"] = parts[0]
            result["Species"] = parts[1]
            result["LifeStage"] = parts[-1]
            return result

        # Genus species lifestage: e.g. "Euphausia triacantha calyptopis"
        if (
            len(parts) == 3
            and parts[0][0].isupper()
            and parts[1][0].islower()
            and parts[2].lower() in LIFE_STAGES
        ):
            result["Genus"] = parts[0]
            result["Species"] = parts[1]
            result["LifeStage"] = parts[2]
            return result

        # Genus (Subgenus) species: e.g. "Acartia (Acartia) danae"
        subgenus_match = re.match(
            r"^([A-Z][a-z]+) \(([A-Z][a-z]+)\) ([a-z]+)$", name
        )
        if subgenus_match:
            result["Genus"] = subgenus_match.group(1)
            result["Species"] = subgenus_match.group(3)
            return result

        # Genus sp.: e.g. "Acartia sp."
        if name.endswith(" sp.") or name.endswith(" sp"):
            genus = re.sub(r" sp\.?$", "", name)
            result["Genus"] = genus
            result["Qualifier"] = "sp"
            return result

        # Genus sp. lifestage: e.g. "Thysanoessa sp. furcilia"
        sp_life_match = re.match(r"^([A-Z][a-z]+) sp\. ([a-z]+)$", name)
        if sp_life_match and sp_life_match.group(2).lower() in LIFE_STAGES:
            result["Genus"] = sp_life_match.group(1)
            result["LifeStage"] = sp_life_match.group(2)
            result["Qualifier"] = "sp"
            return result

        # HigherTaxon lifestage: e.g. "Bryozoa larvae", "Decapoda megalopa"
        if len(parts) == 2 and parts[1].lower() in LIFE_STAGES:
            result["HigherTaxon"] = parts[0]
            result["LifeStage"] = parts[1]
            return result

        # HigherTaxon indet lifestage: e.g. "Calanoida indet (small)"
        indet_life_match = re.match(r"^(.+?) indet \(?([\w]+)\)?$", name)
        if indet_life_match and indet_life_match.group(2).lower() in LIFE_STAGES:
            result["HigherTaxon"] = indet_life_match.group(1)
            result["LifeStage"] = indet_life_match.group(2)
            result["Qualifier"] = "indet"
            return result

        # Taxon lifestage indet: e.g. "Copepoda nauplius indet"
        life_indet_match = re.match(r"^([A-Z][a-z]+) ([a-z]+) indet$", name)
        if life_indet_match and life_indet_match.group(2) in LIFE_STAGES:
            result["HigherTaxon"] = life_indet_match.group(1)
            result["LifeStage"] = life_indet_match.group(2)
            result["Qualifier"] = "indet"
            return result

        # Higher taxon indet: e.g. "Amphipoda indet", "Chaetognatha indet"
        if name.endswith(" indet"):
            taxon = name.replace(" indet", "")
            result["HigherTaxon"] = taxon
            result["Qualifier"] = "indet"
            return result

        # Subspecies or trinomial: e.g. "Clione limacina antarctica"
        if (
            len(parts) == 3
            and parts[0][0].isupper()
            and parts[1][0].islower()
            and parts[2].lower() not in LIFE_STAGES
        ):
            result["Genus"] = parts[0]
            result["Species"] = parts[1]
            return result

        # Species with "var": e.g. "Euphausia similis var armata"
        if " var " in name:
            result["Genus"] = parts[0]
            result["Species"] = parts[1]
            return result

        # Genus + species (default binomial): e.g. "Oithona similis"
        if (
            len(parts) == 2
            and parts[0][0].isupper()
            and parts[1][0].islower()
            and parts[1].lower() not in LIFE_STAGES
        ):
            result["Genus"] = parts[0]
            result["Species"] = parts[1]
            return result

        # Euphausiidae-style family with lifestage indet:
        # e.g. "Euphausiidae calyptopis indet"
        if len(parts) == 3 and parts[2] == "indet":
            result["HigherTaxon"] = parts[0]
            if parts[1].lower() in LIFE_STAGES:
                result["LifeStage"] = parts[1]
            result["Qualifier"] = "indet"
            return result

        # Fallback
        result["RawID"] = name
        return result

    return (parse_taxon_name,)


@app.cell
def _(colnames_species, parse_taxon_name, pd):
    # Parse all species column names
    parsed_taxa = [parse_taxon_name(name) for name in colnames_species]
    parsed_taxa_df = pd.DataFrame(parsed_taxa)
    parsed_taxa_df
    return (parsed_taxa_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. WoRMS Taxonomy Annotation

    Query the World Register of Marine Species (WoRMS) to build a full
    taxonomic hierarchy for each taxon. Results are cached to
    `data/taxon_annotations.csv` to avoid repeated API calls.
    """)
    return


@app.cell(hide_code=True)
def _(Path, pd):
    import pyworms

    TAXONOMY_RANKS = [
        "kingdom",
        "subkingdom",
        "infrakingdom",
        "phylum",
        "subphylum",
        "class",
        "subclass",
        "order",
        "suborder",
        "family",
        "genus",
        "species",
    ]
    _cache_path = Path("data/taxon_annotations.csv")

    def query_worms_taxonomy(parsed_row: dict) -> dict:
        """Query WoRMS for a single parsed taxon and return full hierarchy."""
        # Use None instead of np.nan so columns stay as object dtype
        hierarchy = {r: None for r in TAXONOMY_RANKS}
        hierarchy["ID"] = parsed_row["ID"]
        hierarchy["LifeStage"] = parsed_row.get("LifeStage")
        hierarchy["Qualifier"] = parsed_row.get("Qualifier")

        # Determine search name
        search_name = None
        if "Genus" in parsed_row and pd.notna(parsed_row.get("Genus")):
            if "Species" in parsed_row and pd.notna(parsed_row.get("Species")):
                search_name = f"{parsed_row['Genus']} {parsed_row['Species']}"
            else:
                search_name = parsed_row["Genus"]
        elif "HigherTaxon" in parsed_row and pd.notna(
            parsed_row.get("HigherTaxon")
        ):
            search_name = parsed_row["HigherTaxon"]

        if search_name is None:
            return hierarchy

        try:
            records = pyworms.aphiaRecordsByMatchNames(search_name)
            if not records or not records[0]:
                return hierarchy

            aphia_id = records[0][0].get("AphiaID")
            if aphia_id is None:
                return hierarchy

            # pyworms returns a flat dict with rank names as keys
            classification = pyworms.aphiaClassificationByAphiaID(aphia_id)
            if classification:
                for _rank in TAXONOMY_RANKS:
                    _value = classification.get(_rank)
                    if _value:
                        hierarchy[_rank] = _value

        except Exception:
            pass

        return hierarchy

    def build_taxonomy_table(parsed_df: pd.DataFrame) -> pd.DataFrame:
        """Build full taxonomy table, using cache if available."""
        if _cache_path.exists():
            cached = pd.read_csv(_cache_path)
            if set(parsed_df["ID"]).issubset(set(cached["ID"])):
                return cached

        results = []
        for _idx in range(len(parsed_df)):
            _row = parsed_df.iloc[_idx]
            _result = query_worms_taxonomy(_row.to_dict())
            results.append(_result)

        # Build dataframe with object dtype to allow string assignments
        taxon_df = pd.DataFrame(results)
        for _col in TAXONOMY_RANKS + ["LifeStage", "Qualifier"]:
            if _col in taxon_df.columns:
                taxon_df[_col] = taxon_df[_col].astype(object)

        # Add manual corrections for edge-case taxa
        _manual = _get_manual_corrections()
        for _correction in _manual:
            _mask = taxon_df["ID"] == _correction["ID"]
            if _mask.any():
                for _k, _v in _correction.items():
                    if _k != "ID":
                        taxon_df.loc[_mask, _k] = _v
            else:
                taxon_df = pd.concat(
                    [taxon_df, pd.DataFrame([_correction])], ignore_index=True
                )

        # Remove rows where all rank columns are empty
        _rank_cols = [c for c in TAXONOMY_RANKS if c in taxon_df.columns]
        taxon_df = taxon_df.dropna(subset=_rank_cols, how="all")

        # Cache results
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        taxon_df.to_csv(_cache_path, index=False)

        return taxon_df

    def _get_manual_corrections() -> list[dict]:
        """Manual taxonomy corrections for taxa that WoRMS can't resolve."""
        return [
            {
                "ID": "Appendicularia indet",
                "genus": "Appendicularia",
                "class": "Appendicularia",
                "phylum": "Chordata",
                "kingdom": "Animalia",
                "Qualifier": "indet",
            },
            {
                "ID": "Clione sp.",
                "genus": "Clione",
                "family": "Clionidae",
                "order": "Pteropoda",
                "class": "Gastropoda",
                "phylum": "Mollusca",
                "kingdom": "Animalia",
                "Qualifier": "sp",
            },
            {
                "ID": "Ctenophora indet",
                "phylum": "Ctenophora",
                "kingdom": "Animalia",
                "Qualifier": "indet",
            },
            {
                "ID": "Hyperia sp.",
                "genus": "Hyperia",
                "family": "Hyperiidae",
                "order": "Amphipoda",
                "class": "Malacostraca",
                "phylum": "Arthropoda",
                "kingdom": "Animalia",
                "Qualifier": "sp",
            },
            {
                "ID": "Spongiobranchaea australis",
                "genus": "Spongiobranchaea",
                "species": "australis",
                "family": "Pneumodermatidae",
                "order": "Pteropoda",
                "class": "Gastropoda",
                "phylum": "Mollusca",
                "kingdom": "Animalia",
            },
            {
                "ID": "Themisto sp.",
                "genus": "Themisto",
                "family": "Hyperiidae",
                "order": "Amphipoda",
                "class": "Malacostraca",
                "phylum": "Arthropoda",
                "kingdom": "Animalia",
                "Qualifier": "sp",
            },
        ]

    return (build_taxonomy_table,)


@app.cell
def _(build_taxonomy_table, mo, parsed_taxa_df):
    mo.output.replace(
        mo.md(
            "_Querying WoRMS API (this may take a few minutes on first run)..._"
        )
    )
    taxon_annotations = build_taxonomy_table(parsed_taxa_df)
    mo.output.replace(
        mo.md(f"**Taxonomy table:** {len(taxon_annotations)} taxa annotated")
    )
    taxon_annotations
    return (taxon_annotations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Calculate Indices

    - **Total abundance** corrected by segment length
    - **Salp / Euphausiid dominance index** (positive = salp dominated)
    """)
    return


@app.cell(hide_code=True)
def _(cpr_raw, np, pd):
    # Prepare working dataframe
    abundance_df = cpr_raw.copy()

    # Convert Water_Temperature: replace "-" with NaN
    abundance_df["Water_Temperature"] = pd.to_numeric(
        abundance_df["Water_Temperature"].replace("-", np.nan), errors="coerce"
    )

    # Segment-length corrected total abundance
    abundance_df["Total_Plankton_Corrected"] = pd.to_numeric(
        abundance_df["Total abundance"].replace("-", np.nan), errors="coerce"
    ) / abundance_df["Segment_Length"]

    # Yearly effort
    effort_df = (
        abundance_df.groupby("Year").size().reset_index(name="n_segments")
    )
    return abundance_df, effort_df


@app.cell
def _(abundance_df, effort_df, np, plt):
    # Total abundance time series with effort (static matplotlib)
    fig_abund, (ax_abund, ax_effort) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Scatter with water temperature color
    _wt = abundance_df["Water_Temperature"]
    _valid = _wt.notna()
    _x = abundance_df.loc[_valid, "Year"] + np.random.uniform(
        -0.2, 0.2, size=_valid.sum()
    )
    _sc = ax_abund.scatter(
        _x,
        abundance_df.loc[_valid, "Total_Plankton_Corrected"],
        c=abundance_df.loc[_valid, "Water_Temperature"],
        cmap="plasma",
        s=3,
        alpha=0.5,
    )
    fig_abund.colorbar(_sc, ax=ax_abund, label="Water Temp (°C)", pad=0.01)
    ax_abund.set_ylabel("Total plankton (segment corrected)")
    ax_abund.set_title("Total plankton abundance over time")

    # Effort bar chart
    ax_effort.bar(effort_df["Year"], effort_df["n_segments"], color="lightgrey")
    ax_effort.set_ylabel("# segments")
    ax_effort.set_xlabel("Year")

    plt.tight_layout()
    fig_abund
    return


@app.cell(hide_code=True)
def _(abundance_df, colnames_species, pd, taxon_annotations):
    # Identify Salpidae and Euphausiidae taxa
    salp_ids = taxon_annotations.loc[
        taxon_annotations["family"] == "Salpidae", "ID"
    ].tolist()
    euphausid_ids = taxon_annotations.loc[
        taxon_annotations["family"] == "Euphausiidae", "ID"
    ].tolist()

    # Filter to IDs that exist as columns in the data
    salp_ids = [s for s in salp_ids if s in colnames_species]
    euphausid_ids = [e for e in euphausid_ids if e in colnames_species]

    # Calculate abundances — convert columns to numeric first
    salp_euph_df = abundance_df.copy()

    if salp_ids:
        _salp_numeric = salp_euph_df[salp_ids].apply(
            pd.to_numeric, errors="coerce"
        )
        salp_euph_df["Salp_abundance"] = _salp_numeric.sum(axis=1)
    else:
        salp_euph_df["Salp_abundance"] = 0

    if euphausid_ids:
        _euph_numeric = salp_euph_df[euphausid_ids].apply(
            pd.to_numeric, errors="coerce"
        )
        salp_euph_df["Euph_abundance"] = _euph_numeric.sum(axis=1)
    else:
        salp_euph_df["Euph_abundance"] = 0

    salp_euph_df["SalpEuph_diff"] = (
        salp_euph_df["Salp_abundance"] - salp_euph_df["Euph_abundance"]
    )

    # Annual balance
    annual_balance = (
        salp_euph_df.groupby("Year")
        .agg(SalpEuphIndex=("SalpEuph_diff", "mean"))
        .reset_index()
    )
    return annual_balance, salp_euph_df


@app.cell(hide_code=True)
def _(annual_balance, np, plt, salp_euph_df):
    fig_salp, ax_salp = plt.subplots(figsize=(10, 4))

    _salp_dom = salp_euph_df["SalpEuph_diff"] > 0
    _x_jitter = salp_euph_df["Year"] + np.random.uniform(
        -0.2, 0.2, size=len(salp_euph_df)
    )

    # Euphausiid-dominated (red) and Salp-dominated (blue) points
    ax_salp.scatter(
        _x_jitter[~_salp_dom],
        salp_euph_df.loc[~_salp_dom, "SalpEuph_diff"],
        s=3, alpha=0.5, color="firebrick", label="Euphausiids dominate",
    )
    ax_salp.scatter(
        _x_jitter[_salp_dom],
        salp_euph_df.loc[_salp_dom, "SalpEuph_diff"],
        s=3, alpha=0.5, color="steelblue", label="Salps dominate",
    )

    # Annual balance
    ax_salp.scatter(
        annual_balance["Year"],
        annual_balance["SalpEuphIndex"],
        s=40, color="black", zorder=5, label="Annual balance",
    )

    ax_salp.axhline(0, linestyle="--", color="grey", linewidth=0.8)
    ax_salp.set_ylim(-50, 50)
    ax_salp.set_xlabel("Year")
    ax_salp.set_ylabel("Salp – Euphausiid abundance")
    ax_salp.set_title("Salp – Euphausiid Dominance Index")
    ax_salp.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_salp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Export Results
    """)
    return


@app.cell
def _(Path, abundance_df, colnames_metadata, colnames_species, mo, pd, salp_euph_df, taxon_annotations):
    _output_dir = Path("data")
    _metadata_dir = Path("data/metadata")
    _output_dir.mkdir(parents=True, exist_ok=True)
    _metadata_dir.mkdir(parents=True, exist_ok=True)

    # --- Abundance table: taxa only, indexed by Segment_ID ---
    _abundance_export = abundance_df.set_index("Segment_ID")[colnames_species].copy()
    # Convert all species columns to numeric
    _abundance_export = _abundance_export.apply(pd.to_numeric, errors="coerce")
    _abundance_export.to_csv(_output_dir / "abundance_processed.csv")

    # --- Metadata table: segment info, indexed by Segment_ID ---
    _meta_cols = [
        c for c in colnames_metadata if c in abundance_df.columns
    ] + ["Total_Plankton_Corrected"]
    _metadata_export = abundance_df.set_index("Segment_ID")[_meta_cols].copy()
    _metadata_export.to_csv(_metadata_dir / "segment_metadata.csv")

    # --- Taxonomy annotations ---
    taxon_annotations.to_csv(_metadata_dir / "taxon_annotations.csv", index=False)

    # --- Salp/Euphausiid index table ---
    _salp_euph_export = salp_euph_df.set_index("Segment_ID")[
        [
            "Salp_abundance", "Euph_abundance", "SalpEuph_diff",
        ]
    ].copy()
    _salp_euph_export.to_csv(_output_dir / "salp_euphausiid_index.csv")

    mo.md(f"""
    **Saved outputs:**
    - `data/abundance_processed.csv` ({_abundance_export.shape[0]:,} segments × {_abundance_export.shape[1]} taxa)
    - `data/salp_euphausiid_index.csv` ({len(_salp_euph_export):,} segments)
    - `data/metadata/segment_metadata.csv` ({len(_metadata_export):,} segments — coordinates, env. variables, effort)
    - `data/metadata/taxon_annotations.csv` ({len(taxon_annotations)} taxa — WoRMS taxonomy)
    """)
    return


if __name__ == "__main__":
    app.run()
