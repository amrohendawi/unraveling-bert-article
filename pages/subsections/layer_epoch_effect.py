import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback
from utils import DATA_PATH
from appServer import app

import pandas as pd

tsne_dict = {
    "tsne_hate": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_hate.csv")),
    "tsne_offensive": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_offensive.csv")),
    "tsne_sentiment": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_sentiment.csv")),
}

for dataset in tsne_dict:
    tsne_dict[dataset]['unique_id'] = tsne_dict[dataset].index
    tsne_dict[dataset]['x'] = (tsne_dict[dataset]['x'].round(4) * 10000).astype(int)
    tsne_dict[dataset]['y'] = (tsne_dict[dataset]['y'].round(4) * 10000).astype(int)

tsne_labels_dict = {
    "tsne_hate": ["hate", "no hate"],
    "tsne_offensive": ["offensive", "no offensive"],
    "tsne_sentiment": ["positive", "neutral", "negative"],
}

# for each dataframe in tsne_dict, update the label column with the corresponding labels from tsne_labels_dict
for key, value in tsne_dict.items():
    value['label'] = value['label'].apply(lambda x: tsne_labels_dict[key][x])


def draw_scatter_facet(dataset):
    fig = px.scatter(tsne_dict[dataset], x="x", y="y", color="label",
                     animation_frame="epoch", animation_group="unique_id",
                     facet_col='layer', facet_col_wrap=4,
                     hover_name='label',
                     labels={
                         'y': '',
                         'x': '',
                         'label': '',
                     },
                     hover_data={
                         'epoch': False,
                         'x': False,
                         'y': False,
                         'unique_id': False,
                         'label': False,
                         'layer': False,
                     },
                     opacity=0.8,
                     facet_col_spacing=0,
                     width=800, height=600,
                     )
    fig.update_xaxes(showticklabels=False, showline=True, linewidth=1,
                     linecolor='#4d4d4d', mirror=True)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1,
                     showline=True, linewidth=1,
                     linecolor='#4d4d4d', mirror=True)
    fig.update_traces(hoverinfo='none',
                      marker=dict(line=dict(width=1, color='white')),
                      )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'autosize': True,
        'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0},
    })
    return fig


tsne_figures = {}

# for every dataframe in tsne_dict, draw a scatter plot
for key, value in tsne_dict.items():
    tsne_figures[key] = draw_scatter_facet(key)

content = html.Div([
    dbc.Tooltip(
        "Click to find more about t-SNE",
        target="tsne-anchor",
    ),
    dbc.Offcanvas(
        [
            dbc.Tooltip(
                html.A( "Van der Maaten, Laurens, and Geoffrey Hinton. \"Visualizing data using t-SNE.\" Journal of machine learning research 9, no. 11 (2008).",
                       href="https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl",
                       target="_blank"),
                target="ref-10",
                delay={"show": 0, "hide": 1000},
                placement='left',
                class_name="custom_tooltip",
            ),
            html.Ul([
                html.Li(
                    "The t-SNE algorithm was applied in order to visualize the contextual sequence embeddings for each layer of the transformer model. T-SNE is a nonlinear dimensionality reduction technique that projects nearby points in a high-dimensional manifold closer together in a lower-dimensional space than non-neighboring points. The algorithm consists of two steps: first, assigning pairs of similar high-dimensional points with a higher probability than non-similar pairings; and second, minimizing the Kullback-Leibler divergence between the two computed probability distributions in order to maintain the structure as much as possible with a low projection error rate."),
                html.Li([
                    "In the context of this transformer model, t-SNE was applied to the contextualized embeddings for each layer in order to visualize the patterns and boundaries that the model was learning. The t-SNE algorithm was initialized with a perplexity of 500 in order to preserve the distance between each point and its 500 closest neighbors. This allowed for a reasonable coverage of the global structure. T-SNE was also applied to the training data for each epoch in order to visualize the progression of training. ",
                      html.P("[10]", id="ref-10", className="ref-link")])
            ])
        ],
        id="tsne-canvas",
        title="What is t-SNE?",
        is_open=False,
        placement="end",
        className="offcanvas-class",
    ),



    
    html.Div(
        [

            html.H4(
                "Layer and epoch effect on transferability",
            ),
            html.Br(),
            html.P(
                [
                    """
                      To demonstrate the layer and epoch effect on BERT's transferability, a BERT model is fine-tuned
                      to do classification task on 3 labels. Then, 
                       """,
                    html.A(
                        "t-SNE",
                        id="tsne-anchor",
                        className="toggle-text",
                    ),
                    " is applied on the output to reduce its dimensionality from 768 to 2 dimensions for human readability.",
                ]
            ),
            html.P(
                """
                As observed in different studies, the middle layers of BERT models tend to contain the most syntactic information.
                This is why these layers are the most transferable across tasks.
                Therefore, when using transfer learning with BERT models, it is important to keep this in mind and focus on the
                middle layers.        
                """
            ),
            html.P(
                """
                The higher layers seem to be more effective in this process because they acquire more general understanding.
                The lower layers, on the other hand, seem to be more specialized and only understand specific aspects of the data.
                Additionally, the number of epochs also seems to have an effect on the performance of BERT models.
                """
            ),
            html.P(
                """
                In general, the more epochs, the better the performance. However, this is not always the case, and it is important
                to experiment with different numbers of epochs to find the best results.
                """
            ),
        ],
        id="layer-epoch",
        className="text-box card-component"
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
                "value": "tsne_sentiment",
            },
        ],
        placeholder="Select a dataset",
        value="tsne_sentiment",
    ),
    html.Br(),
    html.Div(
        [
            dcc.Loading(
                dcc.Graph(id='scatter-with-slider',
                          config={"displayModeBar": False},
                          )),
        ], className="card-component", style={"width": "fit-content"}
    ),
    html.P("The results of the following visualization hold two main observations:"),
    html.Ol([
        html.Li(
            "The data points of different classes are highly mixed, and no pattern or discrimination boundaries are yet developed at the beginning of the training loop. As the training progresses, an apparent clustering of the different classes starts to establish itself in some layers."),
        html.Li(
            "The pattern and clustering of the different classes are primarily evident in higher layers of the model."),
        html.Li(
            "The pre-trained Bert model has a low or non-existing understanding of unseen data, but after a proper fine-tuning procedure, it can generalize and adapt to new domains effectively"),
    ]),
    html.Hr(),
])


@callback(

    Output("tsne-canvas", "is_open"),
    Input("tsne-anchor", "n_clicks"),
)
def toggle_text(n1):
    return n1


@app.callback(
    Output('scatter-with-slider', 'figure'),
    Input("dropdown-dataset", "value")
)
def update_figure(dataset):
    return tsne_figures[dataset]
