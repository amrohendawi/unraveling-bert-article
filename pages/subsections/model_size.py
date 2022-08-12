import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from dash import dcc, html, callback, Input, Output
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

canvas = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Compression methods")),
        dbc.ModalBody(
            [
                html.P("""
        BERT models can be compressed using a variety of methods, including quantization, pruning, and distillation.
        In this section, we will focus on the compression methods that offer model size reduction and speedup at inference time.
        """),
                html.Ul([
                    html.Li(
                        [
                            html.H6("Quantization"),
                            html.P("""
                        Quantization is a compression technique that reduces the number of bits used to represent a model.
                        This can be done by reducing the precision of the weights (i.e. using fewer bits to represent
                        each weight) or by using fewer bits to represent the activations (i.e. using fewer bits to
                        represent each neuron output). This technique can offer significant speedups at inference time,
                        as lower precision weights and activations can be processed more quickly.
                        """),
                            dmc.Center(
                                children=[
                                    dmc.Image(
                                        src="/assets/quantization.png", alt="Quantization", caption="Quantization"
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Li(
                        [
                            html.H6("Pruning"),
                            html.P(
                                [
                                    """
                        Pruning refers to the process of identifying and removing less important weights and components
                        from a neural network model. Pruning can sometimes make a model more robust and better-performing.
                        It is also a commonly used method of exploring the lottery ticket hypothesis in neural networks
                        (Frankle and Carbin, 2019), which has also been studied in the context of BERT
                        (Chen et al., 2020b; Prasanna et al., 2020).
                        There are two main types of pruning methods for BERT: unstructured pruning and structured pruning.
                        """,
                                    add_tooltip(references_dict[7]['title'], "8", "ref-7-8",
                                                href=references_dict[7]['href']),
                                ]),
                            html.P("""
                        Unstructured pruning involves pruning individual weights by locating the set of the least
                        important weights in the model. The importance of the weights can be judged by their absolute
                        values, by the gradients, or by some custom-designed measurement. Unstructured pruning could be
                        effective for BERT, given the latter’s massive amount of fully-connected layers. Unstructured
                        pruning methods include magnitude weight pruning, movement-based pruning, and reweighted
                        proximal pruning.
                        """),
                            html.P("""
                        Structured pruning focuses on pruning structured blocks of weights (Li et al., 2020a) or even
                        complete architectural components in the BERT model, by reducing and simplifying certain
                        numerical modules. Structured pruning methods for BERT include attention head pruning, encoder
                        unit pruning, and embedding size pruning.
                        """),
                            dmc.Center(
                                children=[
                                    dmc.Image(
                                        src="/assets/pruning.png", alt="Pruning", caption="Various pruning methods"
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Li(
                        [
                            html.H6("Knowledge Distillation"),
                            html.P("""
                            Knowledge Distillation refers to training a smaller model (called the student) using outputs
                            (from various intermediate functional components) of one or more larger pre-trained models
                            (called the teachers). The flow of information can sometimes be through an intermediate model
                            (commonly known as teaching assistants) (Ding and Yang, 2020; Sun et al., 2020b; Wang et al., 2020c).
                            """),
                            html.P(
                                [
                                    """
                            The main idea behind knowledge distillation is that, by training the student model to mimic
                            the output of the teacher model, the student model can learn the knowledge captured by the 
                            teacher model, and thus, can achieve better performance on the target tasks.
                            """,
                                    add_tooltip(references_dict[7]['title'], "8", "ref-7-4",
                                                href=references_dict[7]['href']),
                                ]
                            ),
                            dmc.Center(
                                children=[
                                    dmc.Image(
                                        src="/assets/distillation.png", alt="distillation",
                                        caption="""
                                        Knowledge distillation. In order to form student models, (a) the encoder
                                        width can be reduced, (b) the number of encoders is reduced, (c) the
                                        BiLSTM can be replaced, (d) the CNN can be replaced, or any combination
                                        of the above can be used.
                                        """
                                    )
                                ]
                            ),
                        ],
                    ),
                    html.Li(
                        [
                            html.H6("Other methods"),
                            html.P(
                                [
                                    """
                                Matrix Decomposition Methods (MDM) focus on decomposing the matrices in the linear layers
                                and attention heads to reduce the matrix multiplication overhead by catering to individual
                                input examples and dynamically changing the amount of computation.
                                """,
                                    add_tooltip(references_dict[7]['title'], "8", "ref-7-5",
                                                href=references_dict[7]['href']),
                                ]
                            ),
                            html.P(
                                [
                                    """
                            Dynamic Inference Acceleration (DIA) is a method that accelerates inference by reducing the
                            computational overhead at inference time
                            """,
                                    add_tooltip(references_dict[7]['title'], "8", "ref-7-6",
                                                href=references_dict[7]['href']),
                                ]
                            ),
                        ],
                    ),
                ]),
                html.P(
                    [
                        """
                        All figures in this section are taken from 
                        """,
                        add_tooltip(references_dict[7]['title'], "8", "ref-7-7",
                                    href=references_dict[7]['href']),
                        "."
                    ]
                )
            ],
        ),
    ],
    id="toggle-compression-offcanvas",
    is_open=False,
    size="lg",
)

text_content = html.Div(
    [
        html.H4("How big should BERT be?"),
        html.Br(),
        html.P("""
        The size of the BERT model has a significant impact on the performance and the time required to complete
        the task. As we know, the pre-trained models are usually computationally expensive and have large memory
        footprint, which is another big obstacle for their deployment in real-world applications. Due to these reasons,
        it is necessary to train a smaller model that is capable of performing well on the target tasks.
        In this section, several scientific studies have been aggregated to reach the following conclusions:
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
        html.P(
            [
                """
                Read more about
                """,
                add_tooltip("Click to read more about model compression methods", "compression methods",
                            "toggle-compression"),
            ]
        )
    ],
    id="model-size"
)

content = html.Div([
    canvas,
    text_content,
    scatter_plot,
    html.Hr()
])


@callback(
    Output("toggle-compression-offcanvas", "is_open"),
    Input("toggle-compression", "n_clicks"),
)
def toggle_tl(n1):
    return n1
