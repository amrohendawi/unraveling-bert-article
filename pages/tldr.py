from dash import html
from utils import textBox

layout = html.Div(
    [
        textBox(
            """
            ### TL;DR
            In this article, we investigate the factors affecting the
            transferability of a BERT model trained on general natural language
            understanding to downstream, more specific tasks. These factors
            include number of layers, number of fine-tuning epochs, dataset
            size, and label noise. We find that, in general, a shallower BERT
            model transfers better than a deeper model, and that a model trained
            with more data and for more epochs transfers better than a model
            trained with less data and for fewer epochs. We also find that label
            noise has a negative effect on transferability.
            """,
        ),
    ],
    id="tldr"
)
