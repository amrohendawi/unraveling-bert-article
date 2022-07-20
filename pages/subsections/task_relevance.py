import plotly.express as px
import plotly.graph_objs as go
import igraph as ig
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate

from utils import DATA_PATH, textBox, read_tasks_nested_tables, df_to_matrix
from appServer import app

import pandas as pd
import json

task_to_task_transfer_learning_res = read_tasks_nested_tables(
    "3_task_to_task_transfer_learning_res", convert_csvs=df_to_matrix)


def create_heatmap(task_class, task_category, dataset_size):
    fig = px.imshow(task_to_task_transfer_learning_res[task_class][task_category][dataset_size],
                    labels={"x": "Target Task", "y": "Source Task"},
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    max_val = task_to_task_transfer_learning_res[task_class][task_category][dataset_size].max().max()
    min_val = task_to_task_transfer_learning_res[task_class][task_category][dataset_size].min().min()

    fig.update_layout(
        title={
            'text': "Task to Task transfer learning results",
            'font_size': 15,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center'},
        coloraxis={
            'colorscale': 'viridis_r',
            'showscale': True,
            'colorbar': dict(
                ticktext=['worst', 'best'],
                tickvals=[min_val+5, max_val-5],
            )
        },

    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
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


def process_graph_data(method, data):
    N = len(data['nodes'])
    L = len(data['links'])
    Edges = [(data['links'][k]['source'], data['links'][k]['target'])
             for k in range(L)]
    # sort Edges by source
    Edges.sort(key=lambda x: x[0])

    #  if edge source and target are the same, then assign it color from colors dict
    # else assign in grey color
    edges_color = []
    for e in Edges:
        if int(e[0] / 11) == int(e[1] / 11):
            if method == 'taskemb':
                edges_color.append('blue' if e[0] < 11 else 'red' if e[0] < 22 else 'green')
            else:
                edges_color.append('red' if e[0] < 11 else 'blue' if e[0] < 22 else 'green')
        else:
            edges_color.append('grey')

    G = ig.Graph(Edges, directed=False)
    labels = []
    group = []
    for node in data['nodes']:
        labels.append(node['name'])
        group.append(node['group'])

    layout = G.layout('kk', dim=3)
    Xn = [layout[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layout[k][1] for k in range(N)]  # y-coordinates
    Zn = [layout[k][2] for k in range(N)]  # z-coordinates
    # from groups and Xn, Yn, Zn, and labels create a dataframe
    df = pd.DataFrame(dict(x=Xn, y=Yn, z=Zn, label=labels, group=group))
    # create a dict from Edges, layout and df
    return {'edges': Edges, 'layout': layout, 'df': df, 'edges_color': edges_color}


network_graph_data = {
    "taskemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TASKEMB_SPACE.json'))),
    "textemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TEXTEMB_SPACE.json'))),
}


def draw_network_graph(method):
    figure_data = network_graph_data_processed[method]

    Xe, Ye, Ze = [], [], []

    df = figure_data['df']
    data = network_graph_data[method]
    for e in figure_data['edges']:
        # x-coordinates of edge ends
        Xe += [figure_data['layout'][e[0]][0], figure_data['layout'][e[1]][0], None]
        Ye += [figure_data['layout'][e[0]][1], figure_data['layout'][e[1]][1], None]
        Ze += [figure_data['layout'][e[0]][2], figure_data['layout'][e[1]][2], None]

    # for every edge create a trace. the line width is the value at data['links'][k]['value']
    traces = []
    groups = {'CR': 1, 'SL': 2, 'QA': 3}
    # group df by group and sort groups alphabetically
    for g in sorted(set(df['group'])):
        traces.append(
            go.Scatter3d(
                x=df[df['group'] == g]['x'],
                y=df[df['group'] == g]['y'],
                z=df[df['group'] == g]['z'],
                mode='markers',
                name=g,
                marker=dict(
                    size=10,
                    color=groups[g],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=df[df['group'] == g]['label'],
                hoverinfo='text'
            )
        )
    for i in range(len(figure_data['edges'])):
        traces.append(
            go.Scatter3d(x=Xe[i * 3:(i + 1) * 3],
                         y=Ye[i * 3:(i + 1) * 3],
                         z=Ze[i * 3:(i + 1) * 3],
                         mode='lines',
                         showlegend=False,
                         line=dict(color=figure_data['edges_color'][i], width=data['links'][i]['value'] * 4),
                         hoverinfo='none',
                         )
        )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    graph_layout = go.Layout(
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=graph_layout)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'autosize': True,
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
        'legend': {'y': 0.5},
    })
    return fig


# for each task, get process graph data
network_graph_data_processed = {}
graph_figs = {}

for key, value in network_graph_data.items():
    network_graph_data_processed[key] = process_graph_data(key, value)
    graph_figs[key] = draw_network_graph(key)

network_graph = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(id="network-graph", className="card-component",
                          style={"width": "auto"},
                          config={"displayModeBar": False},
                          )
            ),
            width=6),
        dbc.Col(
            html.Div(
                id="task_desc",
                className="text-box card-component",
            ), width=6,
        ),
    ],
)

content = html.Div([
    textBox(
        """
            ##### The Importance of Task Relevance
             
            Multitask transfer learning results in improved regularization and transfer compared to single-task learning [2].
            Transferability within the same domain returns better results with few exceptions [2].
            Cross-domain transfer learning also returns better results with few exceptions [2].
            
            BERT models can be fine-tuned for multiple tasks such as natural language understanding (NLU) and natural language generation (NLG) [3].
            BERT's performance on NLU tasks can be improved by fine-tuning on NLG tasks [3].
            This indicates that BERT's representations are general enough to be transferable to different tasks.
            
            Taskonomy can be used to improve the performance of BERT models [4].
            Taskonomy can be used to select the most relevant tasks for fine-tuning a BERT model [4].
            For example, if a BERT model is fine-tuned on a task that is not relevant to the target task, the model's performance on the target task will be worse than if the model was not fine-tuned at all.
            
            Taskonomy can also be used to select the most relevant layers for fine-tuning a BERT model [4].
            For example, if a BERT model is fine-tuned on a task that is not relevant to the target task, the model's performance on the target task will be worse than if the model was not fine-tuned at all.
            
            Overall, taskonomy can be used to improve the performance of BERT models by carefully selecting the tasks and layers for fine-tuning.
            """,
        text_id="task-relevance"
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
    dcc.Dropdown(
        id="dropdown-graph-type",
        searchable=False,
        clearable=False,
        options=[
            {"label": k, "value": k}
            for k in network_graph_data.keys()],
        placeholder="Select an embedding extraction method",
        value=list(network_graph_data.keys())[0],
        className="drop-down-component"
    ),
    network_graph,
]
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


@app.callback(
    Output('network-graph', 'figure'),
    Input("dropdown-graph-type", "value"))
def update_figure(method):
    return graph_figs[method]
