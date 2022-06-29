import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import igraph as ig

from utils import textBox, DATA_PATH
from appServer import app

import json


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


network_graph_data = {
    "taskemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TASKEMB_SPACE.json'))),
    "textemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TEXTEMB_SPACE.json'))),
}


network_graph = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(id="network-graph", className="card-component",)
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
            ## Embeddings Quality
            The quality of the learned embeddings is an important factor in the performance of downstream tasks. If the embeddings are of poor quality, the downstream task will likely suffer.
            There are a few ways to measure the quality of learned embeddings. One is to evaluate the performance of a model that is trained on a supervised task using the learned embeddings as features. Another is to evaluate the performance of a model that is trained on a unsupervised task using the learned embeddings as features.
            BERT models have been shown to produce high-quality embeddings. For example, a study found that a BERT model trained on a large corpus of English text produced embeddings that were better at capturing syntactic and semantic information than word2vec embeddings.
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
],
)


@ app.callback(
    Output('network-graph', 'figure'),
    Input("dropdown-graph-type", "value"))
def update_figure(method):
    data = network_graph_data[method]
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

    graph_layout = go.Layout(
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
    return go.Figure(data=data, layout=graph_layout)
