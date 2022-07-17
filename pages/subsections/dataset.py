import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from utils import textBox, DATA_PATH, df_to_matrix
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
    # get tasks from value
    tasks = value['tasks']
    # append every task to tasks_description dict
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
                html.Div(
                    dcc.Loading(
                        dcc.Graph(
                            id="clickable-heatmap",
                            hoverData={"points": [
                                {"pointNumber": 0}]},
                            config={"displayModeBar": False},
                            style={"padding": "5px 10px"},
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

content = html.Div([
    textBox(
        """
            #### The Effect of Dataset Size
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """,
        text_id="dataset"
    ),
    gen_avg_trans_learning,
])


@app.callback(
    Output("source_target_task_desc", "children"),
    Input("clickable-heatmap", "hoverData")
)
def task_group_info_on_hover(hoverData):
    if hoverData is None:
        raise PreventUpdate
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
    fig = px.imshow(gen_avg_trans_learn_dict[experiment],
                    labels={"x": "Target Task", "y": "Source Task"},
                    contrast_rescaling='minmax',
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    # fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_layout(
        title={
            'text': "Average transfer learning results",
            'font_size': 12,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
        },
        coloraxis={'colorscale': 'viridis'}
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
    })
    return fig
