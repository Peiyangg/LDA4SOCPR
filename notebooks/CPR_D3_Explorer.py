import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import threading
    import http.server
    import socket
    return Path, http, mo, socket, threading


@app.cell
def _(Path, http, socket, threading):
    _figures_dir = str(Path(__file__).resolve().parent.parent / "figures")

    def _find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    _port = _find_free_port()

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=_figures_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    _server = http.server.HTTPServer(("localhost", _port), _QuietHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()

    server_port = _port
    return (server_port,)


@app.cell
def _(mo, server_port):
    mo.md(f"""### Antarctic CPR Data Explorer\nServing visualization on `localhost:{server_port}`""")
    return


@app.cell
def _(mo, server_port):
    mo.Html(
        f'<iframe src="http://localhost:{server_port}/index.html" '
        f'width="100%" height="900" '
        f'style="border:none; border-radius:8px;"></iframe>'
    )
    return


if __name__ == "__main__":
    app.run()
