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
        dbc.Tooltip(
        html.A("Kao, Wei-Tsung, Tsung-Han Wu, Po-Han Chi, Chun-Cheng Hsieh, and Hung-Yi Lee. \"BERT's output layer recognizes all hidden layers? Some Intriguing Phenomena and a simple way to boost BERT.\" arXiv preprint arXiv:2001.09309 (2020).",
               href="https://arxiv.org/abs/2001.09309",
               target="_blank"),
        target="ref-5",
        delay={"show": 0, "hide": 1000},
        placement='top',
        class_name="custom_tooltip",
    ),
     dbc.Tooltip(
        html.A("Michel, Paul, Omer Levy, and Graham Neubig. \"Are sixteen heads really better than one?.\" Advances in neural information processing systems 32 (2019).",
               href="https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html",
               target="_blank"),
        target="ref-6",
        delay={"show": 0, "hide": 1000},
        placement='top',
        class_name="custom_tooltip",
    ),
    dbc.Tooltip(
        html.A("Gordon, Mitchell A., Kevin Duh, and Nicholas Andrews. \"Compressing bert: Studying the effects of weight pruning on transfer learning.\" arXiv preprint arXiv:2002.08307 (2020).",
               href="https://arxiv.org/abs/2002.08307",
               target="_blank"),
        target="ref-7",
        delay={"show": 0, "hide": 1000},
        placement='top',
        class_name="custom_tooltip",
    ),
        html.H4("How big should BERT be?"),
        html.P(
            "The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
        html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
                html.P("[5]", id="ref-5", className="ref-link")]),
        html.P([
            "The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
            html.P("[6]", id="ref-6", className="ref-link")]),  # TODO: elaborate a little on the texts here
        html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
                 html.P("[7]", id="ref-7", className="ref-link")]),
    ],
    id="model-size"
)

content = html.Div([
    text_content,
    html.P([
        dbc.Tooltip(
        html.A("Rogers, A., Kovaleva, O. and Rumshisky, A., 2020. A primer in bertology: What we know about how bert works. Transactions of the Association for Computational Linguistics, 8, pp.842-866.",
               href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT",
               target="_blank"),
        target="ref-8",
        delay={"show": 0, "hide": 1000},
        placement='top',
        class_name="custom_tooltip",
    ),
        "The following table shows many version of Tranformer with many options of comparsion and the related speed up. The values of size, performance and time are against the BERT base in percent",
        html.P("[8]", id="ref-8", className="ref-link")]),

    html.P(
        " It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    scatter_plot,
    html.Hr()
])
