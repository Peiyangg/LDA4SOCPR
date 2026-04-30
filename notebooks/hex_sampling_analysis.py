import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import h3
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import Polygon
    from pathlib import Path

    DATA = Path(__file__).resolve().parent.parent / "data"
    FIGURES = Path(__file__).resolve().parent.parent / "figures"
    return DATA, FIGURES, Polygon, ccrs, cfeature, h3, mcolors, mo, np, pd, plt


@app.cell
def _(mo):
    min_years_slider = mo.ui.slider(
        start=1, stop=30, value=5, step=1,
        label="Minimum years sampled",
    )
    res_toggle = mo.ui.radio(
        options={"Res 2 (~158 km)": 2, "Res 3 (~59 km)": 3},
        value="Res 3 (~59 km)",
        label="H3 Resolution",
    )
    mo.hstack([res_toggle, min_years_slider], gap=2)
    return min_years_slider, res_toggle


@app.cell
def _(DATA, pd, res_toggle):
    resolution = res_toggle.value
    effort = pd.read_csv(DATA / f"hex_effort_res{resolution}.csv", index_col=0)
    metadata = pd.read_csv(DATA / f"hex_metadata_res{resolution}.csv")

    year_cols = [c for c in effort.columns if c.isdigit()]
    effort_years = effort[year_cols]
    n_years_sampled = (effort_years > 0).sum(axis=1)
    total_segments = effort_years.sum(axis=1)

    cell_summary = pd.DataFrame({
        "h3_cell": effort.index,
        "n_years_sampled": n_years_sampled.values,
        "total_segments": total_segments.values.astype(int),
        "first_year": effort_years.apply(
            lambda row: int(min(c for c in year_cols if row[c] > 0)), axis=1
        ).values,
        "last_year": effort_years.apply(
            lambda row: int(max(c for c in year_cols if row[c] > 0)), axis=1
        ).values,
    })

    mean_meta = metadata.groupby("h3_cell").agg(
        center_lat=("center_lat", "mean"),
        center_lon=("center_lon", "mean"),
        mean_sst=("mean_hadisst_sst", "mean"),
        mean_ice=("mean_hadisst_ice", "mean"),
    ).reset_index()
    cell_summary = cell_summary.merge(mean_meta, on="h3_cell", how="left")

    sampled_years_dict = {}
    for cell in effort.index:
        yrs = [int(c) for c in year_cols if effort.at[cell, c] > 0]
        sampled_years_dict[cell] = sorted(yrs)
    cell_summary["sampled_years"] = cell_summary["h3_cell"].map(
        lambda c: sampled_years_dict.get(c, [])
    )

    cell_summary = cell_summary.sort_values("n_years_sampled", ascending=False).reset_index(drop=True)
    return cell_summary, effort_years, resolution, sampled_years_dict, year_cols


@app.cell
def _(cell_summary, min_years_slider, mo):
    min_yr = min_years_slider.value
    frequent = cell_summary[cell_summary["n_years_sampled"] >= min_yr].copy()

    mo.md(f"""
### Hexbins with ≥ {min_yr} years of sampling

**{len(frequent)}** hexbins out of {len(cell_summary)} total ({100*len(frequent)/len(cell_summary):.1f}%)
— covering **{frequent['total_segments'].sum():,}** segments
({100*frequent['total_segments'].sum()/cell_summary['total_segments'].sum():.1f}% of all data)
""")
    return (frequent,)


@app.cell
def _(frequent, mo):
    display_df = frequent[
        ["h3_cell", "n_years_sampled", "total_segments", "first_year", "last_year",
         "center_lat", "center_lon", "mean_sst", "mean_ice"]
    ].copy()
    display_df["center_lat"] = display_df["center_lat"].round(2)
    display_df["center_lon"] = display_df["center_lon"].round(2)
    display_df["mean_sst"] = display_df["mean_sst"].round(1)
    display_df["mean_ice"] = display_df["mean_ice"].round(3)

    mo.ui.table(display_df, selection=None, label="Frequently sampled hexbins")
    return (display_df,)


@app.cell
def _(Polygon, ccrs, cfeature, frequent, h3, mcolors, min_years_slider, np, plt, resolution):
    min_yr_plot = min_years_slider.value

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -80, -35], ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#21262d", edgecolor="#555", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="#0d1117")
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="#333", alpha=0.6)

    norm = mcolors.Normalize(vmin=min_yr_plot, vmax=max(frequent["n_years_sampled"].max(), min_yr_plot + 1))
    cmap = plt.cm.YlOrRd

    for _, row in frequent.iterrows():
        boundary = h3.cell_to_boundary(row["h3_cell"])
        lons = [p[1] for p in boundary] + [boundary[0][1]]
        lats = [p[0] for p in boundary] + [boundary[0][0]]

        needs_split = any(abs(lons[i+1] - lons[i]) > 180 for i in range(len(lons)-1))
        if needs_split:
            continue

        poly = Polygon(list(zip(lons, lats)))
        color = cmap(norm(row["n_years_sampled"]))
        ax.add_geometries(
            [poly], ccrs.PlateCarree(),
            facecolor=color, edgecolor="#58a6ff", linewidth=0.3, alpha=0.8,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label("Years sampled", fontsize=11)

    ax.set_title(
        f"Frequently Sampled Hexbins (≥{min_yr_plot} years, H3 res {resolution})\n"
        f"{len(frequent)} hexbins",
        fontsize=13, fontweight="bold", color="#e6edf3", pad=15,
    )

    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.ax.yaxis.label.set_color("#e6edf3")
    plt.setp(cbar.ax.get_yticklabels(), color="#8b949e")

    plt.tight_layout()
    fig
    return (fig,)


@app.cell
def _(cell_summary, min_years_slider, np, plt):
    min_yr_hist = min_years_slider.value

    fig2, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig2.patch.set_facecolor("#0d1117")

    for ax_item in axes:
        ax_item.set_facecolor("#161b22")
        ax_item.tick_params(colors="#8b949e")
        for spine in ax_item.spines.values():
            spine.set_color("#30363d")

    bins = np.arange(0.5, cell_summary["n_years_sampled"].max() + 1.5, 1)
    axes[0].hist(
        cell_summary["n_years_sampled"], bins=bins,
        color="#3498db", edgecolor="#0d1117", alpha=0.85,
    )
    axes[0].axvline(min_yr_hist, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Threshold = {min_yr_hist}")
    axes[0].set_xlabel("Number of years sampled", color="#e6edf3")
    axes[0].set_ylabel("Number of hexbins", color="#e6edf3")
    axes[0].set_title("Distribution of sampling coverage", color="#e6edf3", fontweight="bold")
    axes[0].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    sorted_n = np.sort(cell_summary["n_years_sampled"].values)[::-1]
    axes[1].plot(range(1, len(sorted_n) + 1), sorted_n, color="#58a6ff", linewidth=1.5)
    axes[1].axhline(min_yr_hist, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Threshold = {min_yr_hist}")
    axes[1].set_xlabel("Hexbin rank", color="#e6edf3")
    axes[1].set_ylabel("Years sampled", color="#e6edf3")
    axes[1].set_title("Ranked sampling coverage", color="#e6edf3", fontweight="bold")
    axes[1].legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3")

    plt.tight_layout()
    fig2
    return


@app.cell
def _(DATA, FIGURES, display_df, frequent, min_years_slider, mo, resolution):
    min_yr_export = min_years_slider.value

    out_path = DATA / f"hex_frequent_res{resolution}_min{min_yr_export}yr.csv"
    export_df = frequent[
        ["h3_cell", "n_years_sampled", "total_segments", "first_year", "last_year",
         "center_lat", "center_lon", "mean_sst", "mean_ice", "sampled_years"]
    ].copy()
    export_df["sampled_years"] = export_df["sampled_years"].apply(
        lambda yrs: ";".join(str(y) for y in yrs)
    )
    export_df.to_csv(out_path, index=False)

    mo.md(f"Exported **{len(export_df)}** hexbins to `{out_path.name}`")
    return


if __name__ == "__main__":
    app.run()
