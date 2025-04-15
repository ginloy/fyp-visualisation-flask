from flask import (
    Flask,
    json,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
    abort,
)

import polars as pl
import polars.selectors as cs
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import scipy.signal as ss
from flask_caching import Cache
from dash_bootstrap_templates import load_figure_template

load_figure_template(["bootstrap", "bootstrap_dark"])  # type: ignore

config = {
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "./cache",
    "CACHE_DEFAULT_TIMEOUT": 0,
}
cache = Cache(config=config)

app = Flask(__name__)
cache.init_app(app)
DF = pl.scan_parquet("./cleaned.pq")
COLS = DF.collect_schema().names()
UNIQUE_ID1 = DF.select("id1").unique().collect().to_series().sort().to_list()
UNIQUE_CONFIGS = DF.select("config").unique().collect().to_series().sort().to_list()
UNIQUE_POS = DF.select("accel_pos").unique().collect().to_series().sort().to_list()

SELECTORS = [
    dict(name="id1", label="Select Type", options=UNIQUE_ID1),
    dict(name="config", label="Select config", options=UNIQUE_CONFIGS),
    dict(name="accel_pos", label="Select Accel Position", options=UNIQUE_POS),
]


def make_graph_with_peaks(df: pl.DataFrame, n: int, title: str):
    def get_peaks(arr: np.ndarray) -> np.ndarray:
        peak_idxs, _ = ss.find_peaks(arr)
        prominences = ss.peak_prominences(arr, peak_idxs)[0]
        top_idxs = np.argsort(prominences)[: -n - 1 : -1]
        return peak_idxs[top_idxs]

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (group, dff) in enumerate(df.group_by("axis", maintain_order=True)):
        axis = group[0]
        peak_idxs = get_peaks(dff["data_mag"].to_numpy())
        x = dff["x"]
        y = dff["data_mag"]
        peaks_x = x[peak_idxs]
        peaks_y = y[peak_idxs]
        fig.add_scatter(
            x=x, y=y, name=axis, legendgroup=f"{i * 2}", line=dict(color=colors[i * 2])
        )
        fig.add_scatter(
            x=peaks_x,
            y=peaks_y,
            name=f"{axis} peaks",
            marker=dict(symbol="x", size=8, color=colors[i * 2 + 1]),
            mode="markers",
            legendgroup=f"{i * 2 + 1}",
        )
        for x, y in zip(peaks_x, peaks_y):
            fig.add_scatter(
                x=[x, x],
                y=[0, y],
                line=dict(dash="dash", color=colors[i * 2 + 1]),
                legendgroup=f"{i * 2 + 1}",
                showlegend=False,
            )
    fig.update_layout(title=dict(text=title))

    return fig


@app.get("/")
def index():
    return redirect(url_for("graphs"))


@app.post("/theme")
def toggle_theme():
    theme = request.cookies.get("theme", "light")
    resp = make_response("", 204)
    if theme == "dark":
        resp.set_cookie("theme", "light")
    else:
        resp.set_cookie("theme", "dark")
    return resp


@app.get("/graphs")
def graphs():
    params = request.cookies.get("graph-params")
    if params is not None and len(request.args) == 0:
        params = json.loads(params)
        return redirect(url_for("graphs", **params))
    resp = make_response(
        render_template(
            "graphs.html",
            params=request.args,
            selectors=SELECTORS,
            form_state=request.args.to_dict(flat=False),
        )
    )
    resp.set_cookie("graph-params", json.dumps(request.args.to_dict(flat=False)))
    return resp


@app.get("/graphs/<id1>")
@cache.cached(query_string=True)
def get_graph(id1: str):
    params = request.args
    config_filter = params.getlist("config")
    pos_filter = params.getlist("accel_pos")
    theme = params.get("theme", "light")
    df = DF.filter(id1=id1)
    if len(config_filter) > 0:
        df = df.filter(pl.col("config").is_in(config_filter))
    if len(pos_filter) > 0:
        df = df.filter(pl.col("accel_pos").is_in(pos_filter))

    if (
        len(config_filter) == 1
        and len(pos_filter) == 1
        and (id1 == "FRF" or id1 == "FFT")
    ):
        fig = make_graph_with_peaks(df.collect(), 5, id1)
    else:
        fig = px.line(
            df.collect(),
            x="x",
            y="data_mag" if id1 != "Time" else "data_real",
            color="axis",
            facet_col="config",
            facet_row="accel_pos",
            render_mode="webgl",
        )
    fig.update_layout(
        dict(title=dict(text=id1), autosize=True),
        template="bootstrap_dark" if theme == "dark" else "bootstrap",
    )
    fig_html = fig.to_html(
        include_plotlyjs=False, full_html=False, default_height="80vh"
    )
    return f"""
        <div class="fade-in w-100">
            {fig_html}
        </div>
    """


@app.get("/data")
def raw_data():
    return render_template("data.html", headers=COLS)


@app.get("/data/table")
@cache.cached(query_string=True)
def get_table_data():
    params = request.args
    draw = int(params.get("draw", 0))
    start = int(params.get("start", 0))
    length = int(params.get("length", 0))
    search_val = params.get("search[value]")
    search_regex = params.get("search[regex]", "").lower() == "true"

    filter_expr = pl.lit(True)
    if search_val:
        filter_expr &= pl.any_horizontal(
            pl.all().cast(pl.String).str.contains(search_val, literal=not search_regex)
        )

    ordering = []
    for i, col in enumerate(COLS):
        search = params.get(f"columns[{i}][search][value]")
        regex = params.get(f"columns[{i}][search][regex]") == "true"
        if search:
            filter_expr &= (
                pl.col(col).cast(pl.String).str.contains(search, literal=not regex)
            )
        order_str = f"order[{i}][column]"
        if order_str in params:
            ordering.append(
                (COLS[int(params[order_str])], params.get(f"order[{i}][dir]") == "desc")
            )
    filtered_df = DF.filter(filter_expr).sort(
        by=[col for col, _ in ordering],
        descending=[desc for _, desc in ordering],
        maintain_order=True,
    )
    data = filtered_df.slice(start).head(length).fill_nan(None).collect()
    data = [list(row) for row in data.rows()]

    return dict(
        draw=draw,
        recordsTotal=DF.select(pl.len()).collect().item(),
        recordsFiltered=filtered_df.select(pl.len()).collect().item(),
        data=data,
    )
