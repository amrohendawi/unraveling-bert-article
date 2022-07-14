import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from utils import textBox, DATA_PATH
from appServer import app

import pandas as pd

tsne_dict = {
    "tsne_hate": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_hate.csv")),
    "tsne_offensive": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_offensive.csv")),
    "tsne_sentiment": pd.read_csv(DATA_PATH.joinpath("tsne_data/tsne_sentiment.csv")),
}

tsne_labels_dict = {
    "tsne_hate": ["hate", "no hate"],
    "tsne_offensive": ["offensive", "no offensive"],
    "tsne_sentiment": ["positive", "neutral", "negative"],
}

def draw_scatter_facet(dataset):
    fig = px.scatter(tsne_dict[dataset], x="x", y="y", color="label",
                     animation_frame="epoch", animation_group="x",
                     facet_col='layer', facet_col_wrap=4,
                     hover_name='label',
                     labels={
                         'layer': '',
                         'y': '',
                         'x': '',
                         'epoch': '',
                         'label': '',
                     },
                     opacity=0.8,
                     facet_col_spacing=0,
                     width=800, height=600,
                     )
    fig.update_xaxes(showticklabels=False, showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1,
                     showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_traces(hoverinfo='none')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'autosize': True,
    })
    return fig

# for each dataframe in tsne_dict, update the label column with the corresponding labels from tsne_labels_dict
for key, value in tsne_dict.items():
    value['label'] = value['label'].apply(lambda x: tsne_labels_dict[key][x])

tsne_figures = {}

# for every dataframe in tsne_dict, draw a scatter plot
for key, value in tsne_dict.items():
    tsne_figures[key] = draw_scatter_facet(key)

content = html.Div([
    textBox(
        """
        #### Layer and epoch effect on transferability
        To demonstrate the layer and epoch effect on BERT's transferability, a BERT model is fine-tuned to do classification task on 3 labels.
        Then, t-SNE is applied on the output to reduce its dimensionality from 768 to 2 dimensions for human readability.

        As observed in different studies, the middle layers of BERT models tend to contain the most syntactic information.
        This is why these layers are the most transferable across tasks.
        Therefore, when using transfer learning with BERT models, it is important to keep this in mind and focus on the
        middle layers.
        
        The higher layers seem to be more effective in this process because they acquire more general understanding.
        The lower layers, on the other hand, seem to be more specialized and only understand specific aspects of the data.
        Additionally, the number of epochs also seems to have an effect on the performance of BERT models.

        Jawahar et al. (2019) found that the lower layers of BERT are more sensitive to lexical information, while the higher layers are more sensitive to syntactic information.
        Hewitt and Manning (2019) had the most success reconstructing syntactic tree depth from the middle BERT layers (6-9 for base-BERT, 14-19 for BERT-large).
        Goldberg (2019) reports the best subject-verb agreement around layers 8-9, and the performance on syntactic probing tasks used by Jawahar et al. (2019) also seems to peak around the middle of the model.
        
        In general, the more epochs, the better the performance. However, this is not always the case, and it is important
        to experiment with different numbers of epochs to find the best results.
            """
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
        className="drop-down-component",
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
    textBox(
        """
                The results of the following visualization hold two main observations:
                1. The data points of different classes are highly mixed, and no pattern or discrimination boundaries are yet developed at the beginning of the training loop. As the training progresses, an apparent clustering of the different classes starts to establish itself in some layers.
                2. The pattern and clustering of the different classes are primarily evident in higher layers of the model.
                The previous observations show that the pre-trained Bert model has a low or non-existing understanding of unseen data, but after a proper fine-tuning procedure, it can generalize and adapt to new domains effectively. Furthermore, the observation shows which layers learn and hold the most discriminating features.
            """
    ),
], id="layer-epoch")


@app.callback(
    Output('scatter-with-slider', 'figure'),
    Input("dropdown-dataset", "value")
)
def update_figure(dataset):
    return tsne_figures[dataset]
