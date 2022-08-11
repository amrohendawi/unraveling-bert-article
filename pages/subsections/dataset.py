import plotly.express as px
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate

from utils import DATA_PATH, df_to_matrix, add_tooltip
from pages.references import references_dict

from appServer import app

import pandas as pd
import json

gen_avg_trans_learn_dict = {
    "full-full": df_to_matrix(
        pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_full_full.csv"))),
    "full-limited": df_to_matrix(
        pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_full_limited.csv"))),
    "limited-limited": df_to_matrix(
        pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_limited_limited.csv"))),
}

tasks_groups_list = json.load(
    open(DATA_PATH.joinpath('33 tasks description.json'))
)

tasks_description = {}
for key, value in tasks_groups_list.items():
    tasks = value['tasks']
    tasks_description.update(tasks)

gen_avg_trans_learning = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Dropdown(
                    id="dropdown-task",
                    searchable=False,
                    clearable=False,
                    options=[
                        {
                            "label": "Full-to-Full dataset",
                            "value": "full-full",
                        },
                        {
                            "label": "Full-to-Limited dataset",
                            "value": "full-limited",
                        },
                        {
                            "label": "Limited-to-Limited dataset",
                            "value": "limited-limited",
                        },
                    ],
                    placeholder="Select an experiment",
                    value="full-full",
                    className="drop-down-component"
                ),
                html.Br(),
                html.Div(
                    dcc.Loading(
                        dcc.Graph(
                            id="clickable-heatmap",
                            hoverData={"points": [
                                {"pointNumber": 0}]},
                            config={"displayModeBar": False},
                        )
                    ), className="card-component",
                    style={"width": "auto"},
                )
            ], width=4,
        ),
        # A text box to show the current task description
        dbc.Col(
            html.Div(
                id="source_target_task_desc",
                className="text-box card-component",
            ), width=8
        ),
    ]
)

canvases = [
    dbc.Offcanvas(
        [
            html.P(
                [
                    """
                        The authors examined the impact of data size on transfer learning by performing experiments in three
                        data regimes: FULL → FULL, FULL → LIMITED, and LIMITED → LIMITED.
                    """,
                    add_tooltip(references_dict[3]['title'],
                                "4",
                                "ref-4-12",
                                href=references_dict[3]['href'],
                                ),
                ]
            ),
            html.Ul([
                html.Li(
                    """
                    In FULL training, all training data for the associated task is used to fine-tune the model.
                    """
                ),
                html.Li(
                    """
                    For the LIMITED setting, an artificial limit of 1K training examples was imposed by randomly
                    selecting them without replacement.
                    """
                ),
            ]),
            html.P(
                """
                The authors measured the impact of transfer learning by computing the relative transfer gain given a
                source task s and target task t. For example, if a baseline model that is directly tuned on the target
                dataset achieves a performance of pt, whereas a transferred model achieves a performance of ps→t, the
                relative transfer gain is defined as: gs→t = (ps→t − pt) / pt.
                """
            ),
        ],
        id="dataset-canvas",
        title="How the results were obtained",
        is_open=False,
        placement="end",
    ),
]

text_content = html.Div(
    [
        html.H4("The Effect of Dataset Size"),
        html.Br(),
        html.P(
            """
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to
            generalize to other tasks. There is some evidence that  effect of source and target dataset size can also
            improve a knowledge transformation.
            """),
        html.P(
            """
            The effect of source and target dataset size can impact the performance of a model when transferring knowledge:
            """
        ),
        html.Ul([
            html.Li(
                """
                If the source dataset is too small, the model may not be able to learn the relevant features for the target dataset.
                """
            ),
            html.Li(
                """
                If the source dataset is too large, the model may overfit on the source dataset and not be able
                to generalize to the target dataset.
                """
            ),
            html.Li(
                """
                 The size of the target dataset can also impact the performance of the model.
                """
            ),
        ]),
        html.P(
            [
                """
                These results obtained from Vu Tu et al's study
                """,
                add_tooltip(references_dict[3]['title'],
                            "4",
                            "ref-4-1",
                            href=references_dict[3]['href'],
                            ),
                """
                regarding the effect of data set size on performance are illustrated
                in the heatmaps below.
                """,
                add_tooltip("Click to read more in-depth description about this experiment.", "more about",
                            "toggle-dataset-canvas"),
            ]
        ),
    ],
    id="dataset"
)

content = html.Div([
    *canvases,
    text_content,
    gen_avg_trans_learning,
    # TODO: speake about the methodolgy, how the result has been achived and maybe in each set up show a text at the bottem overviews the results
    html.Hr()
],
)


@app.callback(
    Output("source_target_task_desc", "children"),
    Input("clickable-heatmap", "hoverData")
)
def task_group_info_on_hover(hoverData):
    if hoverData is None or "x" not in hoverData["points"][0].keys():
        return dmc.Alert(
            title="Usage",
            color="teal",
            children=[
                html.P(
                    """
                    The three set-ups shown represent different combinations of source and target data set sizes.
                    """
                ),
                html.P(
                    "Each cell represents a domain of tasks. Hover over a heatmap cell to view a task description!"
                ),
            ]
        )
    try:
        source_id = hoverData["points"][0]["x"]
        target_id = hoverData["points"][0]["y"]
        src_desc = tasks_groups_list[source_id]['description']
        src_name = tasks_groups_list[source_id]['category']
        target_desc = tasks_groups_list[target_id]['description']
        target_name = tasks_groups_list[target_id]['category']

        return html.Div([
            html.H4(src_name),
            html.P(src_desc),
            html.H4(target_name) if source_id != target_id else "",
            html.P(target_desc) if source_id != target_id else "",
        ])

    except Exception:
        raise PreventUpdate


@app.callback(
    Output('clickable-heatmap', 'figure'),
    Input("dropdown-task", "value"))
def update_figure(experiment):
    max_val = gen_avg_trans_learn_dict[experiment].max().max()
    min_val = gen_avg_trans_learn_dict[experiment].min().min()
    fig = px.imshow(gen_avg_trans_learn_dict[experiment],
                    labels={"x": "Target Task", "y": "Source Task"},
                    color_continuous_scale=px.colors.diverging.Geyser_r,
                    )
    fig.update_traces(xgap=1, selector=dict(type='heatmap'))
    fig.update_traces(ygap=1, selector=dict(type='heatmap'))

    fig.update_coloraxes(colorbar_orientation="h")
    fig.update_layout(
        title={
            'text': "<b>Average transfer learning results</b>",
            'font_size': 13,
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
        },
        coloraxis={
            'colorbar': dict(
                ticktext=['worst', 'best'],
                tickvals=[min_val + 10, max_val - 10],
                xpad=0,
                y=0.74,
            ),
        },
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
    })
    return fig


@callback(
    Output("dataset-canvas", "is_open"),
    Input("toggle-dataset-canvas", "n_clicks"),
)
def toggle_text(n1):
    return n1
