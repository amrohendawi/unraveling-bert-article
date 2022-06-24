from pydoc import classname
from appServer import app
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go

from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
import igraph as ig

import pandas as pd
import pathlib
import json

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

tsne_dict = {
    "tsne_hate": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_hate.csv")),
    "tsne_offensive": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_offensive.csv")),
    "tsne_sentiment_multi": pd.read_csv(DATA_PATH.joinpath("tsne_bert_base_cased_sentiment_multi.csv")),
}


def df_to_matrix(df):
    df_matrix = pd.DataFrame(columns=df['sourceTask'].unique())
    for _, row in df.iterrows():
        df_matrix.loc[row['destinationTask'], row['sourceTask']] = row['value']
    return df_matrix


# Convert the content of 3_task_to_task_transfer_learning_res folder to a json file with nested structure
# recursive function
def get_task_list(folder):
    new_dict = {}
    for dir in DATA_PATH.joinpath(folder).iterdir():
        # if the directory is a directory, call the function again
        if dir.is_dir():
            new_dict[dir.name] = get_task_list(folder + "/" + dir.name)
        # if the directory is a csv file, read the file and add the content to the dictionary
        if dir.is_file() and dir.suffix == ".csv":
            # read csv an convert it to json
            df = df_to_matrix(pd.read_csv(dir))
            new_dict[dir.name] = df
    return new_dict


task_to_task_transfer_learning_res = get_task_list(
    "3_task_to_task_transfer_learning_res")

gen_avg_trans_learn_dict = {
    "full-full": df_to_matrix(pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_full_full.csv"))),
    "full-limited": df_to_matrix(pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_full_limited.csv"))),
    "limited-limited": df_to_matrix(pd.read_csv(DATA_PATH.joinpath("1_general__avg_transfer_learning_res_table/gen_limited_limited.csv"))),
}

tasks_groups_list = json.load(
    open(DATA_PATH.joinpath('33 tasks description.json')))

tasks_description = {}
for key, value in tasks_groups_list.items():
    # get tasks from value
    tasks = value['tasks']
    # append every task to tasks_description dict
    tasks_description.update(tasks)


def get_cytoscope_graph():
    data = json.load(open(DATA_PATH.joinpath('TASKEMB_SPACE.json')))
    data2 = json.load(open(DATA_PATH.joinpath('TASKEMB_SPACE_cyto.json')))
    Edges, labels, group, layt, Xn, Yn, Zn = process_graph_data(data)
    points = data['nodes']
    edges = data['links']
    # concatenate points and edges into one list
    nodes = []
    for point in points:
        # append every point in the following format : {'data': {'id': 'id', 'label': 'group'}, 'position': {'x': Xn, 'y': Yn}},
        nodes.append(
            {'data': {'id': point['name'], 'label': point['group']}, 'position': {
                'x': Xn[points.index(point)]*100, 'y': Yn[points.index(point)]*100}}
        )
    for edge in data2['links']:
        nodes.append(
            {'data': {'source': edge['source'], 'target': edge['target']}})
    return nodes


network_graph_data = {
    "taskemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TASKEMB_SPACE.json'))),
    "textemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TEXTEMB_SPACE.json'))),
}


def get_network_graph():
    # read local json file TASKEMB_SPACE.json
    data = network_graph_data['textemb']
    Edges, labels, group, layt, Xn, Yn, Zn = process_graph_data(data)
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        # x-coordinates of edge ends
        Xe += [layt[e[0]][0], layt[e[1]][0], None]
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

    trace1 = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=dict(color='rgb(125,125,125)', width=1),
                          hoverinfo='none'
                          )

    trace2 = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode='markers',
                          name='actors',
                          marker=dict(symbol='circle',
                                      size=8,
                                      color=group,
                                      colorscale='Viridis',
                                      line=dict(
                                          color='rgb(50,50,50)', width=0.5)
                                      ),
                          text=labels,
                          hoverinfo='text'
                          )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        # title=" A 3D visualization of the task spaces TASKEMB captures task similarity (the two part-of-speech tagging tasks are interconnected despite their domain dissimilarity).",
        # width=500,
        # height=500,
        # showlegend=True,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                    size=14
                )
            )
        ],)

    data = [trace1, trace2]
    return go.Figure(data=data, layout=layout)


def process_graph_data(data):
    N = len(data['nodes'])
    L = len(data['links'])
    Edges = [(data['links'][k]['source'], data['links'][k]['target'])
             for k in range(L)]

    G = ig.Graph(Edges, directed=False)
    labels = []
    group = []

    groups = {'CR': 1, 'SL': 2, 'QA': 3}

    for node in data['nodes']:
        labels.append(node['name'])
        group.append(groups[node['group']])

    layt = G.layout('kk', dim=3)
    Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]  # y-coordinates
    Zn = [layt[k][2] for k in range(N)]  # z-coordinates
    return Edges, labels, group, layt, Xn, Yn, Zn


# Text sections
factors_section = html.Div([dcc.Markdown(
    """
        # Factors
        In this section we discuss the main factors that affect the
        transferability of BERT. These factors are the number of layers,
        the number of fine-tuning epochs, the dataset size, and the label
        noise.
        """.replace(
        "  ", ""
    )
)],
    className="text-box card-component")
fine_tuning_section = html.Div([
    dcc.Markdown(
        """
            # Fine-tuning
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.

            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """.replace(
            "  ", ""
        ),
    ),
],
    className="text-box card-component")

layer_epoch_effect = html.Div([dcc.Markdown(
    """
        # Layer and epoch effect on transferability
        As observed in different studies, the middle layers of BERT models tend to contain the most syntactic information.
        This is likely due to the fact that these layers are the most transferable across tasks.
        Therefore, when using transfer learning with BERT models, it is important to keep this in mind and focus on the
        middle layers. Additionally, the number of epochs also seems to have an effect on the performance of BERT models.
        
        In general, the more epochs, the better the performance. However, this is not always the case, and it is important
        to experiment with different numbers of epochs to find the best results.

        Jawahar et al. (2019) found that the lower layers of BERT are more sensitive to lexical information, while the higher layers are more sensitive to syntactic information.
        Hewitt and Manning (2019) had the most success reconstructing syntactic tree depth from the middle BERT layers (6-9 for base-BERT, 14-19 for BERT-large).
        Goldberg (2019) reports the best subject-verb agreement around layers 8-9, and the performance on syntactic probing tasks used by Jawahar et al. (2019) also seems to peak around the middle of the model.
            """.replace(
        "  ", ""
    ), className="text-box card-component"
),
    dcc.Dropdown(
    id="dropdown-dataset",
    searchable=False,
    clearable=False,
    options=[
        {
            "label": "BERT base cased hate",
            "value": "tsne_hate",
        },
        {
            "label": "BERT base cased offensive",
            "value": "tsne_offensive",
        },
        {
            "label": "BERT base cased sentiment",
            "value": "tsne_sentiment_multi",
        },
    ],
    placeholder="Select a dataset",
    value="tsne_sentiment_multi",
    className="drop-down-component",
),
    html.Div(
    [
        dcc.Loading(
            dcc.Graph(id='scatter-with-slider', style={'width': '100vh', 'height': '100vh'})),
    ], className="card-component",
),
    html.Div([dcc.Markdown(
        """
                The results of the following visualization hold two main observations:

                1. The data points of different classes are highly mixed, and no pattern or discrimination boundaries are yet developed at the beginning of the training loop. As the training progresses, an apparent clustering of the different classes starts to establish itself in some layers.
                2. The pattern and clustering of the different classes are primarily evident in higher layers of the model.

                The previous observations show that the pre-trained Bert model has a low or non-existing understanding of unseen data, but after a proper fine-tuning procedure, it can generalize and adapt to new domains effectively. Furthermore, the observation shows which layers learn and hold the most discriminating features.

            """.replace(
            "  ", ""
        ),
    )],
    className="text-box card-component"),
],
)

dataset_section = html.Div([dcc.Markdown(
    """
            # Dataset
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """.replace(
        "  ", ""
    ),
)],
    className="text-box card-component")

task_to_task_trans_learning = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Dropdown(
                    id="dropdown-class",
                    searchable=False,
                    clearable=False,
                    # the keys from task_to_task_trans_learning_res
                    options=[
                        {"label": k, "value": k}
                        for k in task_to_task_transfer_learning_res.keys()],
                    placeholder="Select a class",
                    value=list(task_to_task_transfer_learning_res.keys())[0],
                    className="drop-down-component"
                ),
                dcc.Dropdown(
                    id="dropdown-task-category",
                    searchable=False,
                    clearable=False,
                    # the keys from task_to_task_trans_learning_res
                    options=[
                        {"label": k, "value": k}
                        for k in task_to_task_transfer_learning_res['inclass'].keys()],
                    placeholder="Select a task category",
                    value=list(
                        task_to_task_transfer_learning_res['inclass'].keys())[0],
                    className="drop-down-component"
                ),
                dcc.Dropdown(
                    id="dropdown-dataset-size",
                    searchable=False,
                    clearable=False,
                    # the keys from task_to_task_trans_learning_res
                    options=[
                        {"label": k, "value": k}
                        for k in task_to_task_transfer_learning_res['inclass']['1_classification_inclass'].keys()],
                    placeholder="Select a dataset size",
                    value=list(
                        task_to_task_transfer_learning_res['inclass']['1_classification_inclass'].keys())[0],
                    className="drop-down-component"
                ),
                html.Div(
                    dcc.Loading(
                        dcc.Graph(
                            id="clickable-heatmap2",
                            hoverData={"points": [
                                {"pointNumber": 0}]},
                            config={"displayModeBar": False},
                            style={"padding": "5px 10px"},
                        )
                    ), className="card-component"
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

task_similarity_section = html.Div([
    dcc.Markdown(
        """
            # Task Similarity
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """.replace(
            "  ", ""
        ), className="text-box card-component"
    ),
    task_to_task_trans_learning,
    dcc.Markdown(
        """
            It has been shown that fine-tuning a model on a specific task can improve its transferability to other tasks.
            This is because the model has been specifically optimized for the task at hand and is better able to generalize to other tasks.
            There is also some evidence that fine-tuning can improve a model's ability to transfer to other domains. For example, a model
            that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            
            In the limited setting, the mean and standard deviation across 20 random restarts are reported.
            In the out-of-class transfer results, the orange-colored row Baseline shows the results of fine-tuning BERT on target tasks without any intermediate fine-tuning.
            Positive transfers are shown in blue and the best results are highlighted in bold (blue). These results suggest that fine-tuning can improve a model's ability to transfer to other tasks and domains.
            """.replace(
            "  ", ""
        ), className="text-box card-component"
    ),
],
)

network_graph = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(figure=get_network_graph(),
                          id="network-graph", className="card-component",)
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

embeddings_quality_section = html.Div([dcc.Markdown(
    """
            # Embeddings Quality
            The quality of the learned embeddings is an important factor in the performance of downstream tasks. If the embeddings are of poor quality, the downstream task will likely suffer.
            There are a few ways to measure the quality of learned embeddings. One is to evaluate the performance of a model that is trained on a supervised task using the learned embeddings as features. Another is to evaluate the performance of a model that is trained on a unsupervised task using the learned embeddings as features.
            BERT models have been shown to produce high-quality embeddings. For example, a study found that a BERT model trained on a large corpus of English text produced embeddings that were better at capturing syntactic and semantic information than word2vec embeddings.
            """.replace(
        "  ", ""
    ),
    className="text-box card-component"
),
    dcc.Dropdown(
    id="dropdown-graph-type",
    searchable=False,
    clearable=False,
    # the keys from task_to_task_trans_learning_res
    options=[
        {"label": k, "value": k}
        for k in network_graph_data.keys()],
    placeholder="Select an embedding extraction method",
    value=list(network_graph_data.keys())[0],
    className="drop-down-component"
),
    network_graph,
],
)

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
                    ), className="card-component"
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

layout = html.Div([
    html.Div(
        [
            factors_section,
            fine_tuning_section,
            layer_epoch_effect,
            dataset_section,
            gen_avg_trans_learning,
            task_similarity_section,
            embeddings_quality_section,
            # html.Div([
            #     cyto.Cytoscape(
            #         id='cytoscape-elements-basic',
            #         layout={'name': 'preset'},
            #         style={'width': '100%', 'height': '400px'},
            #         elements=get_cytoscope_graph()
            #     )
            # ]),
        ],
        id="page",
        className="row",
    ),
])


@ app.callback(
    Output('scatter-with-slider', 'figure'),
    Input("dropdown-dataset", "value"))
def update_figure(dataset):
    fig = px.scatter(tsne_dict[dataset], x="x", y="y", color="label",
                     animation_frame="epoch", animation_group="x",
                     log_x=True, size_max=100, facet_col='layer', facet_col_wrap=4)

    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_coloraxes(showscale=False)

    return fig


@ app.callback(
    Output('clickable-heatmap', 'figure'),
    Input("dropdown-task", "value"))
def update_figure(experiment):
    fig = px.imshow(gen_avg_trans_learn_dict[experiment],
                    labels={"x": "Target Task", "y": "Source Task"},
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    fig.update_layout(coloraxis_colorbar_y=-0.001)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_layout(
        title={
            'text': "General average transfer learning results on <br> " + experiment,
            'font_size': 15,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center'})
    return fig


@ app.callback(
    Output('clickable-heatmap2', 'figure'),
    [
        Input("dropdown-class", "value"),
        Input("dropdown-task-category", "value"),
        Input("dropdown-dataset-size", "value"),
    ])
def update_figure(task_class, task_category, dataset_size):
    fig = px.imshow(task_to_task_transfer_learning_res[task_class][task_category][dataset_size],
                    labels={"x": "Target Task", "y": "Source Task"},
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    fig.update_layout(coloraxis_colorbar_y=-0.001)
    # fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_layout(
        title={
            'text': "Task to Task transfer learning results",
            'font_size': 15,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center'})
    return fig


@ app.callback(
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

    except Exception as error:
        print(error)
        raise PreventUpdate


@ app.callback(
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
        # print(error)
        raise PreventUpdate


@ app.callback(
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
        print(error)
        raise PreventUpdate
