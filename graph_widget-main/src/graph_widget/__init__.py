import importlib.metadata
from importlib.resources import files

import anywidget
import traitlets

try:
    __version__ = importlib.metadata.version("graph_widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

class ForceGraphWidget(anywidget.AnyWidget):
    # _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    # _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    _esm = files("graph_widget").joinpath("static/widget.js")
    _css = files("graph_widget").joinpath("static/widget.css")
    data = traitlets.Dict().tag(sync=True)
    repulsion = traitlets.Int().tag(sync=True)
    node_scale = traitlets.Float().tag(sync=True)
    node_size_feature = traitlets.Unicode().tag(sync=True)
    colour_feature = traitlets.Unicode().tag(sync=True)
    colour_scale_type = traitlets.Unicode().tag(sync=True)
    selected_ids = traitlets.List([]).tag(sync=True)
    width = traitlets.Int().tag(sync=True)
    height = traitlets.Int().tag(sync=True)