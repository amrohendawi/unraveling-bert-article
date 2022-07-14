import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from utils import DATA_PATH, textBox, read_tasks_nested_tables, df_to_matrix
from appServer import app

import json

task_to_task_transfer_learning_res = read_tasks_nested_tables(
    "3_task_to_task_transfer_learning_res", convert_csvs=df_to_matrix)


def create_heatmap(task_class, task_category, dataset_size):
    fig = px.imshow(task_to_task_transfer_learning_res[task_class][task_category][dataset_size],
                    labels={"x": "Target Task", "y": "Source Task"},
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    fig.update_layout(
        title={
            'text': "Task to Task transfer learning results",
            'font_size': 15,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center'})
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig


heatmaps = {}
for task_class in task_to_task_transfer_learning_res:
    heatmaps[task_class] = {}
    for task_category in task_to_task_transfer_learning_res[task_class]:
        heatmaps[task_class][task_category] = {}
        for dataset_size in task_to_task_transfer_learning_res[task_class][task_category]:
            heatmaps[task_class][task_category][dataset_size] = create_heatmap(task_class, task_category, dataset_size)

tasks_groups_list = json.load(
    open(DATA_PATH.joinpath('33 tasks description.json')))

tasks_description = {}
for key, value in tasks_groups_list.items():
    # get tasks from value
    tasks = value['tasks']
    # append every task to tasks_description dict
    tasks_description.update(tasks)

task_to_task_trans_learning = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown-class",
                        searchable=False,
                        clearable=False,
                        # the keys from task_to_task_trans_learning_res
                        options=[
                            {"label": k, "value": k}
                            for k in task_to_task_transfer_learning_res.keys()],
                        placeholder="Select a class",
                        value=list(
                            task_to_task_transfer_learning_res.keys())[0],
                        className="drop-down-component"
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown-task-category",
                        searchable=False,
                        clearable=False,
                        placeholder="Select a task category",
                        className="drop-down-component"
                    ),
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown-dataset-size",
                        searchable=False,
                        clearable=False,
                        placeholder="Select a dataset size",
                        className="drop-down-component"
                    ),
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            dcc.Loading(
                                dcc.Graph(
                                    id="clickable-heatmap2",
                                    hoverData={"points": [
                                        {"pointNumber": 0}]},
                                    config={"displayModeBar": False},
                                    style={"padding": "5px 10px"},
                                )
                            ), className="card-component",
                            style={"width": "auto"},
                        )
                    ], width=6,
                ),
                # A text box to show the current task description
                dbc.Col(
                    html.Div(
                        id="source_target_task_desc2",
                        className="text-box card-component",
                    ), width=6
                ),
            ]
        )
    ]
)

content = html.Div([
    textBox(
        """
            #### Task Similarity
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """
    ),
    task_to_task_trans_learning,
    textBox(
        """
            It has been shown that fine-tuning a model on a specific task can improve its transferability to other tasks.
            This is because the model has been specifically optimized for the task at hand and is better able to generalize to other tasks.
            There is also some evidence that fine-tuning can improve a model's ability to transfer to other domains. For example, a model
            that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            
            In the limited setting, the mean and standard deviation across 20 random restarts are reported.
            In the out-of-class transfer results, the orange-colored row Baseline shows the results of fine-tuning BERT on target tasks without any intermediate fine-tuning.
            Positive transfers are shown in blue and the best results are highlighted in bold (blue). These results suggest that fine-tuning can improve a model's ability to transfer to other tasks and domains.
            """
    ),
], id="task-similarity"
)


@app.callback(
    Output("task_desc", "children"),
    Input("network-graph", "hoverData")
)
def task_info_on_hover(hoverData):
    if hoverData is None:
        raise PreventUpdate
    try:
        task_id = hoverData["points"][0]["text"]
        task_desc = tasks_description[task_id]

        return html.Div([
            html.H4(task_id),
            html.P(task_desc)
        ])

    except Exception as error:
        raise PreventUpdate


@app.callback(
    [
        Output('dropdown-task-category', 'options'),
        Output('dropdown-task-category', 'value'),
    ],
    Input('dropdown-class', 'value')
)
def update_tasks_category_dropdown(task_class):
    return [{'label': i, 'value': i} for i in task_to_task_transfer_learning_res[task_class].keys()], \
           list(task_to_task_transfer_learning_res[task_class].keys())[0]


@app.callback(
    [
        Output('dropdown-dataset-size', 'options'),
        Output('dropdown-dataset-size', 'value'),
    ],
    [
        Input('dropdown-class', 'value'),
        Input('dropdown-task-category', 'value'),
    ]
)
def update_tasks_category_dropdown(task_class, task_category):
    return [{'label': i, 'value': i} for i in task_to_task_transfer_learning_res[task_class][task_category].keys()], \
           list(task_to_task_transfer_learning_res[task_class][task_category].keys())[0]

@app.callback(
    Output('clickable-heatmap2', 'figure'),
    [
        Input("dropdown-class", "value"),
        Input("dropdown-task-category", "value"),
        Input("dropdown-dataset-size", "value"),
    ])
def update_figure(task_class, task_category, dataset_size):
    return heatmaps[task_class][task_category][dataset_size]


@app.callback(
    Output("source_target_task_desc2", "children"),
    Input("clickable-heatmap2", "hoverData")
)
def task_info_on_hover(hoverData):
    if hoverData is None:
        raise PreventUpdate
    try:
        source_id = hoverData["points"][0]["x"]
        target_id = hoverData["points"][0]["y"]
        src_desc = tasks_description[source_id]
        target_desc = tasks_description[target_id]

        return html.Div([
            html.H4(source_id),
            html.P(src_desc),
            html.H4(target_id) if source_id != target_id else "",
            html.P(target_desc) if source_id != target_id else "",
        ])

    except Exception as error:
        raise PreventUpdate
