import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px

from utils import DATA_PATH
import pandas as pd

scatter_plot_data = pd.read_csv(DATA_PATH.joinpath("bert_compression_results_modified.csv"))

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
    "title": "Language Model Compression/Performance Comparison",
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
        dbc.Tooltip(
            html.A(
                "Kao, Wei-Tsung, Tsung-Han Wu, Po-Han Chi, Chun-Cheng Hsieh, and Hung-Yi Lee. \"BERT's output layer recognizes all hidden layers? Some Intriguing Phenomena and a simple way to boost BERT.\" arXiv preprint arXiv:2001.09309 (2020).",
                href="https://arxiv.org/abs/2001.09309",
                target="_blank"),
            target="ref-5",
        ),
        dbc.Tooltip(
            html.A(
                "Michel, Paul, Omer Levy, and Graham Neubig. \"Are sixteen heads really better than one?.\" Advances in neural information processing systems 32 (2019).",
                href="https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html",
                target="_blank"),
            target="ref-6",
        ),
        dbc.Tooltip(
            "Gordon, Mitchell A., Kevin Duh, and Nicholas Andrews. \"Compressing bert: Studying the effects of weight pruning on transfer learning.\" arXiv preprint arXiv:2002.08307 (2020).",
            target="ref-7",
        ),
        html.H4("How big should BERT be?"),
        html.Br(),
        html.P(
            "The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
        html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
                html.P("[5]", id="ref-5", className="ref-link")]),
        html.P([
            "The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
            html.P("[6]", id="ref-6", className="ref-link")]),  # TODO: elaborate a little on the texts here
        html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
                html.P("[7]", id="ref-7")]),
    ],
    id="model-size"
)

content = html.Div([
    text_content,
    html.P([
        dbc.Tooltip(
            "Rogers, A., Kovaleva, O. and Rumshisky, A., 2020. A primer in bertology: What we know about how bert works. Transactions of the Association for Computational Linguistics, 8, pp.842-866.",
            target="ref-8",
        ),
        "The following table shows many version of Tranformer with many options of comparsion and the related speed up. The values of size, performance and time are against the BERT base in percent",
        html.P("[8]", id="ref-8", className="ref-link")]),

    html.P(
        " It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    scatter_plot,
    html.Hr()
])
