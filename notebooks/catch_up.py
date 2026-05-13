import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    test = pd.read_csv('data/abundance_processed.csv', index_col=0)
    return (test,)


@app.cell
def _(test):
    test
    return


@app.cell
def _(pd):
    meta = pd.read_csv('data/metadata/segment_metadata.csv', index_col=0)
    return (meta,)


@app.cell
def _(meta):
    meta
    return


@app.cell
def _(pd):
    fronts = pd.read_csv('/Users/huopeiyang/Downloads/Fronts_of_the_Antarctic_Circumpolar_Current_-_GIS_data/csv/antarctic_circumpolar_current_fronts.csv')
    return (fronts,)


@app.cell
def _(fronts):
    fronts
    return


@app.cell
def _(pd):
    import xarray as xr
    import numpy as np

    ds = xr.open_dataset('data/Polar_Front_weekly.nc', decode_times=False)

    # Convert time_stamp (yyyymmdd floats) to proper dates
    dates = pd.to_datetime(ds['time_stamp'].values.astype(int).astype(str))

    # PFw shape: (612 weeks, 1440 longitudes)
    # Each value = latitude of the Polar Front at that longitude/week
    lons = ds['longitude'].values        # 0.125 to 359.875
    pf_lat = ds['PFw'].values            # (612, 1440) — latitude values

    # Example: get the PF latitude at longitude 180° for the first week
    lon_idx = np.argmin(np.abs(lons - 180.0))
    print(f"Week {dates[0]}: PF at lon 180° is at lat {pf_lat[0, lon_idx]:.2f}°")
    return dates, lon_idx, lons, pf_lat, xr


@app.cell
def _(dates):
    dates
    return


@app.cell
def _(lons):
    lons
    return


@app.cell
def _(pf_lat):
    pf_lat
    return


@app.cell
def _(lon_idx):
    lon_idx
    return


@app.cell
def _(xr):
    sst_ds = xr.open_dataset("/Users/huopeiyang/Downloads/sst.day.mean.2023.nc")
    print(sst_ds)                       # shows variables, dims, coords
    sst_jan18 = sst_ds.sst.sel(time="2023-01-18")
    sst_jan18.plot()

    # point sample (lon is 0..360 in this file)
    lon, lat = 78.7, -64.8
    val = sst_ds.sst.sel(time="2023-01-18",
                     lat=lat, lon=lon if lon >= 0 else lon + 360,
                     method="nearest").item()
    return sst_jan18, val


@app.cell
def _(sst_jan18):
    sst_jan18
    return


@app.cell
def _(val):
    val
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
