import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import igraph as ig

from utils import textBox, DATA_PATH
from appServer import app
import pandas as pd
import json


def process_graph_data(data):
    N = len(data['nodes'])
    L = len(data['links'])
    Edges = [(data['links'][k]['source'], data['links'][k]['target'])
             for k in range(L)]

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
    return Edges, layout, df


network_graph_data = {
    "taskemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TASKEMB_SPACE.json'))),
    "textemb": json.load(open(DATA_PATH.joinpath('text_task_embedding_space/TEXTEMB_SPACE.json'))),
}

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
], id="embeddings-quality"
)


@app.callback(
    Output('network-graph', 'figure'),
    Input("dropdown-graph-type", "value"))
def update_figure(method):
    data = network_graph_data[method]
    Edges, layt, df = process_graph_data(data)
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        # x-coordinates of edge ends
        Xe += [layt[e[0]][0], layt[e[1]][0], None]
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

    # for every edge create a trace. the line width is the value at data['links'][k]['value']
    traces = []
    groups = {'CR': 1, 'SL': 2, 'QA': 3}
    # group df by group
    for g in set(df['group']):
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
    for i in range(len(Edges)):
        traces.append(
            go.Scatter3d(x=Xe[i * 3:(i + 1) * 3],
                         y=Ye[i * 3:(i + 1) * 3],
                         z=Ze[i * 3:(i + 1) * 3],
                         mode='lines',
                         showlegend=False,
                         line=dict(color='rgb(125,125,125)', width=data['links'][i]['value'] * 4),
                         hoverinfo='none',
                         # hide label and legend
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
