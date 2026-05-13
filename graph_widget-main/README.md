# graph_widget
An anywidget implementation of force-graph (https://github.com/vasturiano/force-graph). It includes a brush (Cmd-drag).


## Installation

```sh
pip install graph_widget
```

or with [uv](https://github.com/astral-sh/uv):

```sh
uv add graph_widget
```

## Example
An example marimo script (available as `example.py` => run with `uv run marimo edit example.py`):

```python
import graph_widget
import pandas as pd
import json
import requests

url = "https://raw.githubusercontent.com/observablehq/sample-datasets/refs/heads/main/miserables.json"
response = requests.get(url)
data = response.json()

repulsion_slider = mo.ui.slider(
    start=-100, stop=500, step=10, value=1, debounce=False, label="Repulsion"
)
node_scale_slider = mo.ui.slider(
    start=1, stop=20, step=1, value=3, debounce=True, label="Node scale"
)
colour_feature_dropdown = mo.ui.dropdown(
    options=list(data["nodes"][3].keys()), value="kind", label="Colour by"
)
colour_scale_type_radio = mo.ui.radio(options=["diverging", "sequential"], value="diverging", label="Colour scale type")

data_graph = mo.ui.anywidget(
    graph_widget.ForceGraphWidget(
        data=data,
        repulsion=2,
        node_scale=2,
        colour_feature="",
        colour_scale_type=""
    )
)

data_graph.repulsion = repulsion_slider.value
data_graph.node_scale = node_scale_slider.value
data_graph.colour_feature = colour_feature_dropdown.value
data_graph.colour_scale_type= colour_scale_type_radio.value

mo.hstack([data_graph,
    mo.vstack([
        repulsion_slider,
        node_scale_slider,
        colour_feature_dropdown,
        colour_scale_type_radio])], justify="start")

selected = data_graph.selected_ids
```

### Arguments
- `repulsion`
- `node_scale`
- `colour_feature`
- `colour_scale_type`: can be empty string, or `"diverging"`

## Development

We recommend using [uv](https://github.com/astral-sh/uv) for development.
It will automatically manage virtual environments and dependencies for you.

```sh
uv run marimo edit example.py
```

Changes made in `src/graph_widget/static/` will be reflected in the notebook.
