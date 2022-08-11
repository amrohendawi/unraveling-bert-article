import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from dash import dcc, html
import plotly.express as px
from pages.references import references_dict

from utils import DATA_PATH, add_tooltip
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
    "title": "<b>Language Model Compression/Performance Comparison</b>",
    "title_y": 0.97,
    "title_x": 0.5,
    "font_size": 10,
    "legend_title_text": '<b>Method</b>',
    'margin': {'l': 0, 'r': 0},
    "legend": {
        "orientation": "h",
        "yanchor": 'top',
        "xanchor": 'center',
        "y": 1.08,
        "x": 0.5,
    },
    "xaxis": {
        "title": "Compression Factor",
        "type": "log",
        "title_font": {"size": 14},
    },
    "yaxis": {
        "title": "Performance",
        "type": "log",
        "title_font": {"size": 14},
    },
    'plot_bgcolor': 'rgba(89, 151, 129, 0.25)',
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id='model-size-graph',
                            figure=scatter_plot_fig,
                            config={"displayModeBar": False},
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dmc.Alert(
                            title="Usage",
                            color="teal",
                            children=[
                                dmc.List(
                                    [
                                        dmc.ListItem(
                                            "The white circles represent the original size before compression."
                                        ),
                                        dmc.ListItem(
                                            [dmc.Kbd("Click"), " to remove a legend from the graph"],
                                        ),
                                        dmc.ListItem(
                                            [dmc.Kbd("Double click"), " a legend to focus on it."],
                                        ),
                                        dmc.ListItem(
                                            [dmc.Kbd("Select"), " an area of the graph to zoom in."],
                                        ),
                                        dmc.ListItem(
                                            [dmc.Kbd("Double click"), " on the graph to reset zoom."],
                                        ),
                                    ],
                                    style={"font-size": "14px"},
                                ),
                            ],
                            style={"margin-top": "5rem"},
                        ),
                    ],
                    width=4,
                ),
            ],
            className="card-component"
        ),
        dmc.Alert(
            title="Observations",
            color="dark",
            variant="outline",
            children=[
                dmc.List(
                    [
                        dmc.ListItem(
                            [
                                "BERT can be efficiently compressed with little to no accuracy loss ",
                                add_tooltip(references_dict[7]['title'], "8", "ref-7-3",
                                            href=references_dict[7]['href']),
                                "."
                            ]
                        ),
                        dmc.ListItem(
                            "After a certain threshold in compression factor, the performance starts to drop rapidly.",
                        ),
                        dmc.ListItem(
                            [
                                "In some cases, the performance has improved after compression as in ",
                                add_tooltip(references_dict[8]['title'], "ALBERT-xxlarge", "ref-8-1",
                                            href=references_dict[8]['href']),
                                "."
                            ]
                        ),
                    ],
                ),
            ],
            style={"width": "fit-content", "margin": "0 auto"},
        ),
    ])

text_content = html.Div(
    [
        html.H4("How big should BERT be?"),
        html.Br(),
        html.P("""
        The size of the BERT model has a significant impact on the performance and the time required to complete
        the task. In this section, several scientific studies have been aggregated to reach the following conclusions:
        """),
        html.Ul([
            html.Li(
                [
                    """
                    Too many BERT heads and layers can be harmful to the performance of downstream tasks.
                    """,
                    add_tooltip(references_dict[4]['title'], "5", "ref-5-1", href=references_dict[4]['href']),
                ]
            ),
            html.Li(
                [
                    """
                    The disabling of certain heads in the architecture had a positive effect on machine translation and
                    abstractive summarization.
                    """,
                    add_tooltip(references_dict[5]['title'], "6", "ref-6-1", href=references_dict[5]['href']),
                ]
            ),
            html.Li(
                [
                    """
                    30-40 percent of weights can be pruned without any impact on downstream tasks.
                    """,
                    add_tooltip(references_dict[6]['title'], "7", "ref-7-1", href=references_dict[6]['href']),
                ]
            ),
            html.Li(
                [
                    """
                    It is often best to train a larger model and then compress it. The benefits of compression are that it
                    can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make
                    BERT more transferable.
                    """,
                    add_tooltip(references_dict[7]['title'], "8", "ref-7-2", href=references_dict[7]['href']),
                ]
            ),
        ]),
    ],
    id="model-size"
)

content = html.Div([
    text_content,
    scatter_plot,
    html.Hr()
])
