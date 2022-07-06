import dash_bootstrap_components as dbc
from dash import html
from utils import textBox

layout = dbc.Row(
    [
        dbc.Col(
            html.Iframe(
                src='https://my.spline.design/bertarchitecture3d-0acb3f660b62e4d2136948931cb8676a/',
                width='100%',
                height='100%'
            ),
            width=6,
        ),
        dbc.Col(
            textBox(
                """
                ## TL;DR
                In this article, we investigate the factors affecting the
                transferability of a BERT model trained on general natural language
                understanding to downstream, more specific tasks. These factors
                include number of layers, number of fine-tuning epochs, dataset
                size, and label noise. We find that, in general, a shallower BERT
                model transfers better than a deeper model, and that a model trained
                with more data and for more epochs transfers better than a model
                trained with less data and for fewer epochs. We also find that label
                noise has a negative effect on transferability.
                """
            ),
            width=6,
        ),

    ],
    id="tldr"
)
