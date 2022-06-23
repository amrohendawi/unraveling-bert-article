# -*- coding: utf-8 -*-
# Import required libraries
import pathlib
from dash.dependencies import Input, Output
from dash import dcc, html, Input, Output
# import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from appServer import app

from pages import abstract, introduction, factors, directions, conclusion, references

server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "10rem",
    "padding": "2rem",
}

# dictionary for the headlines from TEXTS
HEADLINES = {
    0: "Abstract",
    10: "Introduction",
    20: "Factors",
    24: "Fine-tuning",
    28: "Dataset",
    32: "Task Similarity",
    36: "Embeddings Quality",
    40: "Directions & Further Research",
    # 4: "Acknowledgements",
    50: "Conclusion",
    60: "References",
}

TEXTS = {
    0: """
    # Abstract
    In this article, we investigate the factors affecting the
    transferability of a BERT model trained on general natural language
    understanding to downstream, more specific tasks. These factors
    include number of layers, number of fine-tuning epochs, dataset
    size, and label noise. We find that, in general, a shallower BERT
    model transfers better than a deeper model, and that a model trained
    with more data and for more epochs transfers better than a model
    trained with less data and for fewer epochs. We also find that label
    noise has a negative effect on transferability.
    """.replace(
        "  ", ""
    ),
    10: """
    # Introduction
        Since their introduction in 2017, Transformers (Vaswani et al.,
        2017) have taken NLP by storm, offering enhanced parallelization and
        better modeling of long-range dependencies. The best known
        Transformer-based model is BERT (Devlin et al., 2019); it obtained
        state-of-the-art results in numerous benchmarks and is still a
        must-have baseline.

        BERT is a deep learning model that has been shown to be effective
        for a range of natural language understanding tasks. However, it is
        not clear how well BERT transfers to more specific tasks. In this
        article, we investigate the factors that affect the transferability
        of BERT. These factors include the number of layers, the number of
        fine-tuning epochs, the dataset size, and the label noise.

        We first consider the linguistic aspects of it, such as the number
        of layers, the number of fine-tuning epochs, and the dataset size.
        We find that, in general, a shallower BERT model transfers better
        than a deeper model, and that a model trained with more data and for
        more epochs transfers better than a model trained with less data and
        for fewer epochs. We also consider the effect of label noise on
        transferability. We find that label noise has a negative effect on
        transferability.

        In conclusion, we find that the transferability of BERT is affected
        by a number of factors, including the number of layers, the number
        of fine-tuning epochs, the dataset size, and the label noise.
    """.replace(
        "  ", ""
    ),
    20: """
    # Factors
    In this section we discuss the main factors that affect the
    transferability of BERT. These factors are the number of layers,
    the number of fine-tuning epochs, the dataset size, and the label
    noise.
    """.replace(
        "  ", ""
    ),
    24: """
    # Fine-tuning
    When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
    This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
    
    There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
    For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
    However, it is not clear how much of an improvement fine-tuning provides in this case.
    """.replace(
        "  ", ""
    ),
    28: """
    # Dataset
    When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
    This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
    
    There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
    For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
    However, it is not clear how much of an improvement fine-tuning provides in this case.
    """.replace(
        "  ", ""
    ),
    32: """
    # Task Similarity
    When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
    This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
    
    There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
    For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
    However, it is not clear how much of an improvement fine-tuning provides in this case.
    """.replace(
        "  ", ""
    ),
    36: """
    # Embeddings Quality
    When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
    This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.
    
    There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
    For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
    However, it is not clear how much of an improvement fine-tuning provides in this case.
    """.replace(
        "  ", ""
    ),
    40: """
    # Directions and Further Research
    There is still much research that needs to be done in order to fully understand how fine-tuning affects BERT's transferability.
    In particular, it would be interesting to compare the transferability of models that are fine-tuned on different tasks and domains.
    Additionally, it would be helpful to investigate how different amounts of fine-tuning affect transferability.
    """.replace(
        "  ", ""
    ),
    50: """
    # Conclusion
    Fine-tuning can improve a model's transferability, although the extent of the improvement may vary depending on the task and domain.
    Other factors like the dataset and the model's size  can also affect transferability.
    """.replace(
        "  ", ""
    ),
    60: """
    # References
    1. https://arxiv.org/pdf/1906.01083.pdf
    2. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
    3. https://huggingface.co/transformers/pretrained_models.html#bert
    4. https://medium.com/@jonathan_hui/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
    """.replace(
        "  ", ""
    ),
}

navBar = html.Div(
        [
            html.H2("Content", className="display-4", style= {"font-style": "bold"}),
            html.Hr(),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Slider(
                                min=0,
                                max=max(HEADLINES.keys()),
                                value=0,
                                step=1,
                                marks={i: {'label': HEADLINES[i], 'style':{
                                    'transform': 'scaleY(-1)', 'fontSize': '16px' if i % 10 == 0 else '14',
                                    'font-weight': 'bold' if i % 10 == 0 else 'normal',
                                    'width': 'max-content'}} for i in HEADLINES.keys()},
                                id="slider-vertical",
                                vertical=True,
                            )
                        ],
                        className="timeline-slider-vertical",
                        style={'transform': 'scaleY(-1)'}
                    ),
                ],
                style={"padding": "15px 15px 15px 15px"},
            )
        ],
        style=SIDEBAR_STYLE,
    )

body = html.Div([
    html.Div(
        [
            dcc.Markdown(
                """
                    # Unraveling BERT's Transferability
                    """.replace(
                    "  ", ""
                ),
                className="title",
                style={"textAlign": "center"},
            ),
        ],
        className="row",
        style={"marginTop": "25px"},
    ),
    html.Div(id='page-content')
], style=CONTENT_STYLE)

################### App layout ###################
app.layout = html.Div(
    dbc.Row(
        [
            dcc.Location(id="url"),
            dcc.Store(id="click-output"),
            dbc.Col(navBar, width=3, style={
                "position": "fixed",
                "z-index": "1",
                "top": "0",
                "left": "0",
                "overflow-x": "hidden",
                }),
            dbc.Col(body, width=9),
        ],
        justify="center",
    ),
)

@app.callback(Output('page-content', 'children'),
              [Input("slider-vertical", "value")])
def display_page(value):
    if value == 0:
        return abstract.layout
    elif value == 10:
        return introduction.layout
    elif 20 <= value < 40:
        return factors.layout
    elif value == 40:
        return directions.layout
    elif value == 50:
        return conclusion.layout
    elif value == 60:
        return references.layout
    else:
        """ test """

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(debug=False)
