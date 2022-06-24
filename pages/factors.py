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

# convert loaded dataframes to matrix shape


def df_to_matrix(df):
    df_matrix = pd.DataFrame(columns=df['sourceTask'].unique())
    for _, row in df.iterrows():
        df_matrix.loc[row['destinationTask'], row['sourceTask']] = row['value']
    return df_matrix


transfer_learning_dict = {
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


def get_network_graph():
    # read local json file TASKEMB_SPACE.json
    data = json.load(open(DATA_PATH.joinpath('TASKEMB_SPACE.json')))

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
                                      size=6,
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
fine_tuning_section = html.Div([dcc.Markdown(
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
)],
    className="text-box card-component")
layer_epoch_effect = html.Div([dcc.Markdown(
    """
                # BERT contextual sequence embeddings visualization
                Transformer models are preeminent due to their potential to incorporate context during learning, primarily attributable to self-attention coupled with high-dimensional vectors reflecting multiple degrees of information abstraction embedded within the model's layers. Moreover, high-dimensional vectors cannot be inspected or analyzed to validate the learning process and the type of captured information. However, utilizing nonlinear dimensionality reduction techniques, the dimensionality of the embeddings can be projected onto a lower-dimensional space, thereby reducing the embeddings to an interpretable human subspace (two or three-dimensional cartesian coordinate system). One prominent method is t-distributed stochastic neighbor embedding (t-SNE), which projects nearby points in a high-dimensional manifold closer together in a lower-dimensional space than non-neighboring points. t-SNE consists of two phases to achieve its objectives.
                The first step includes assigning pairs of similar high-dimensional points with a higher probability than non-similar pairings. The second step is identical to the first only for lower-dimensional points, and then the Kullback–Leibler divergence between the two computed probability distributions is minimized to maintain the structure as much as possible with a low projection error rate. The resulting projections can vary depending on the starting parameters of t-SNE. By projecting the contextualized embeddings of each layer of a given transformer using t-SNE, the following component illustrates each layer's ability to uncover patterns and discrimination boundaries on a given dataset for a particular task along with the progression of training epochs. Additionally, the component is modular to cover all changeable components and inputs, such as the model, task, dataset, number of training epochs, various hyperparameters, and layers to inspect.
                The following example applies the tool to the sequence classification domain by fine-tuning the pre-trained Bert-base-uncased model on binary and multi-class text classification tasks taken from the TweetEval dataset (hate, offensive, and sentiment). Moreover, around 2000 training samples are taken from each dataset and then preprocessed initially by encoding noise-generating factors such as URLs, hashtags, usernames, and emojis, removing extra spaces and lowering the text. The preprocessed text is then tokenized by Bert's fast tokenizer provided by the hugging face framework and loaded in a batch sampler to feed the model throughout the training process dynamically. The model is trained for four epochs using a batch size of 16 and a learning rate of 1e-5, and leveraging the maximum sequence length of 512, shorter or longer input samples are padded or truncated, respectively. Furthermore, openTSNE is deployed and used for the dimensionality reduction phase due to its highly efficient and parallel implementation. Additionally, t-SNE is initialized with a perplexity of 500, which preserve the distance between each point and its 500 closest neighbors. A higher number of maintained neighbors allows covering the global structure reasonably. t-SNE also provides the possibility of applying PCA to the input samples as an initial reduction technique followed by the actual procedure. At the completion of each training epoch, the initialized t-SNE function is then applied to the averaged embedding from across all non-masked tokens in the sequence of each training sample along each model layer resulting in a (5, 12, 2000, 2) matrix containing all 2d projections for each layer across each training epoch. Each layer is then depicted in a different figure, and each data sample is mapped to a specific color corresponding to the respective class.
            """.replace(
        "  ", ""
    ),
)],
    className="text-box card-component")
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
task_similarity_section = html.Div([dcc.Markdown(
    """
            # Task Similarity
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
embeddings_quality_section = html.Div([dcc.Markdown(
    """
            # Embeddings Quality
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

task_to_task_transferability = dbc.Row(
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
                ),
                html.Div(
                    dcc.Loading(
                        dcc.Graph(
                            id="clickable-heatmap",
                            hoverData={"points": [
                                {"pointNumber": 0}]},
                            config={"displayModeBar": False},
                            style={"padding": "0px 10px"},
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

layout = html.Div([
    html.Div(
        [
            factors_section,
            fine_tuning_section,
            layer_epoch_effect,
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
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(id='scatter-with-slider', style={'width': '100vh', 'height': '100vh'})),
                ],
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
            dataset_section,
            task_to_task_transferability,
            task_similarity_section,
            network_graph,
            # html.Div([
            #     cyto.Cytoscape(
            #         id='cytoscape-elements-basic',
            #         layout={'name': 'preset'},
            #         style={'width': '100%', 'height': '400px'},
            #         elements=get_cytoscope_graph()
            #     )
            # ]),
            embeddings_quality_section,
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
    # fig.update_layout(paper_bgcolor='#f8f9fa')

    fig.update_coloraxes(showscale=False)

    return fig


@ app.callback(
    Output('clickable-heatmap', 'figure'),
    Input("dropdown-task", "value"))
def update_figure(experiment):
    fig = px.imshow(transfer_learning_dict[experiment],
                    labels={"x": "Target Task", "y": "Source Task"},
                    )
    fig.update_coloraxes(colorbar_orientation="h")
    fig.update_layout(coloraxis_colorbar_y=-0.001)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
    fig.update_layout(
        title={
            'text': "Transfer learning results on <br> " + experiment,
            'font_size': 15,
            'y': 0.9,
            'x': 0.6,
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
