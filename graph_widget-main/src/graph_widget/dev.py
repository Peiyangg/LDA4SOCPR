import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _():
    import anywidget
    import traitlets

    class ForceGraphWidget(anywidget.AnyWidget):
        _esm = "graph.js"
        _css = "graph.css"
        data = traitlets.Dict().tag(sync=True)
        repulsion = traitlets.Int().tag(sync=True)
        node_scale = traitlets.Int().tag(sync=True)
        colour_feature = traitlets.Unicode().tag(sync=True)
        selected_ids = traitlets.List([]).tag(sync=True)
    return ForceGraphWidget, anywidget, traitlets


@app.cell
def _():
    import json
    import requests

    url = "https://raw.githubusercontent.com/observablehq/sample-datasets/refs/heads/main/miserables.json"
    response = requests.get(url)
    data = response.json()
    return data, json, requests, response, url


@app.cell
def _(data, pd):
    df = pd.DataFrame(data["nodes"])
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    repulsion_slider = mo.ui.slider(
        start=-200, stop=10000, step=10, value=1, debounce=False, label="Repulsion"
    )
    node_scale_slider = mo.ui.slider(
        start=1, stop=500, step=1, value=20, debounce=True, label="Node scale"
    )
    return node_scale_slider, repulsion_slider


@app.cell
def _(data, mo):
    colour_feature_selector = mo.ui.dropdown(
        list(data["nodes"][0].keys()), value="id", label="Colour feature"
    )
    return (colour_feature_selector,)


@app.cell
def _(
    ForceGraphWidget,
    colour_feature_selector,
    data,
    mo,
    node_scale_slider,
    repulsion_slider,
):
    data_graph = mo.ui.anywidget(
        ForceGraphWidget(
            data=data,
            repulsion=repulsion_slider.value,
            node_scale=node_scale_slider.value,
            colour_feature=colour_feature_selector.value,
        )
    )
    return (data_graph,)


@app.cell
def _(
    colour_feature_selector,
    data_graph,
    mo,
    node_scale_slider,
    repulsion_slider,
):
    plot = mo.hstack([ data_graph,
                mo.vstack([
                    repulsion_slider,
                    node_scale_slider,
                    colour_feature_selector])])
    return (plot,)


@app.cell
def _(plot):
    plot
    return


@app.cell
def _(data_graph):
    selected = data_graph.selected_ids
    return (selected,)


@app.cell
def _(selected):
    selected
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
