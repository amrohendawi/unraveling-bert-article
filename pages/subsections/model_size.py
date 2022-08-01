import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go

from utils import DATA_PATH
from appServer import app
import pandas as pd



scatter_plot_data = pd.read_csv(DATA_PATH.joinpath(
    "bert_compression_results_modified.csv"))


scatter_plot_fig = px.scatter(
    scatter_plot_data,
    x="Compression",
    y="Performance",
    color="Method",
    size="#Parameters",
    trendline="ols",
    hover_data={
        "Name": True,
        "Performance": True,
        "Speedup": True,
        "#Parameters": True,
        "Compression": True,
        "Method": False,
    },

)

scatter_plot_fig.update_layout({
    "title": "BERT Compression Performance",
    "xaxis": {
        "title": "Compression Factor",
        "type": "log",

    },
    "yaxis": {
        "title": "Performance",
        "type": "log",
    },
    'plot_bgcolor': 'rgba(77, 137, 110, 0.22)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'autosize': True,
})

scatter_plot_fig.update_yaxes(tickformat='.0%')
# convert x axis to a factor scale such as 1x, 2x, 3x, 4x, etc.
# scatter_plot_fig.update_xaxes(
#     #     for any number convert it to a string of format "1x", "2x", "3x", "4x", etc.
#     tickvals=scatter_plot_data['Compression'],
#     ticktext=scatter_plot_data['Compression'].apply(lambda x: str(x) + "x"),
# #     keep it logarithmic
#     type="log",
#
# )

# update traces marker line width by the compression factor for each trace
scatter_plot_fig.update_traces(marker=dict(
    line=dict(width=scatter_plot_data['Compression'])
))

scatter_plot = html.Div(
    [
        dcc.Graph(
            id='model-size-graph',
            figure=scatter_plot_fig,
            config={"displayModeBar": False},
        ),
    ],
    className="card-component", style={"width": "fit-content"}
)

text_content = html.Div(
    [
        html.H4("How big should BERT be?"),
        html.P(
            "The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
        html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
                html.A("[5]", id="t5-ref", href="#references")]),
        html.P([
            "The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
            html.A("[6", id="t6-ref", href="#references")]),  # TODO: elaborate a little on the texts here
        html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
                html.A("[7]", id="t7-ref", href="#references")]),
    ],
    id="model-size"
)

content = html.Div([
    text_content,
    html.P([
        "The following table shows many version of Tranformer with many options of comparsion and the related speed up. The values of size, performance and time are against the BERT base in percent",
        html.A("[8]", id="t8-ref", href="#references")]),

    html.P(
        " It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    scatter_plot,
    html.Hr()
])
