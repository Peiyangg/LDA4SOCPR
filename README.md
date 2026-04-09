# LDA for Southern Ocean CPR Data

**Live interactive map:** [https://peiyangg.github.io/LDA4SOCPR/](https://peiyangg.github.io/LDA4SOCPR/)

---

## Overview

This repository documents an exploratory analysis of Antarctic Continuous Plankton Recorder (CPR) survey data, combining interactive visualisation with topic-modelling approaches to understand spatial and compositional patterns in Southern Ocean plankton communities.

## Interactive Map

An interactive map was built to explore the raw CPR dataset directly in the browser — no local setup required. The map lets you:

- Browse CPR tow locations across the Southern Ocean
- Inspect sample metadata and species composition
- Toggle between LDA topic assignments (k = 3, 7, 10, 16) to compare model outputs spatially

Access it here: [https://peiyangg.github.io/LDA4SOCPR/](https://peiyangg.github.io/LDA4SOCPR/)

## Analysis Approach

### Exploratory Data Analysis (EDA)

Initial EDA was carried out to characterise the dataset — sample distributions over time and space, species richness, abundance patterns, and the influence of environmental covariates. This stage helped identify data quality issues and guided preprocessing decisions.

### Latent Dirichlet Allocation (LDA)

LDA topic modelling was applied to the species-abundance matrix to discover latent community assemblages. Several values of *k* (number of topics) were tested. Results are still in the **exploration stage** — the spatial coherence of topics and their ecological interpretation are being assessed, and no final model has been selected.

## Repository Structure

```
figures/          # Interactive map (index.html) and associated JS/data files
notebooks/        # Analysis scripts (EDA, LDA, preprocessing, UMAP/TDA)
LDA_gensim/       # Gensim-based LDA experiments
data/             # Processed data files
papers/           # Reference literature
```

## Acknowledgements

Data downloading and taxonomic preprocessing follow the workflow described in:

> Deschepper (2026). *WoRMS tutorial for Antarctic CPR data.*
> [https://thesnakeguy.github.io/worrms_tutorial/worrms_antarctic_tutorial.html](https://thesnakeguy.github.io/worrms_tutorial/worrms_antarctic_tutorial.html)

Deschepper is also a collaborator on this project.

## Status

> This project is in active exploration. Results and visualisations are preliminary.
