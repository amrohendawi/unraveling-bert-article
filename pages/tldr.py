import dash_bootstrap_components as dbc
from dash import html, Input, Output, State
from utils import textBox
from appServer import app

layout = html.Div(
    [
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
            """,
        ),
        # collapsible section
        dbc.Button(
            "Open collapse",
            id="collapse-button",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody("This content is hidden in the collapse")),
            id="collapse",
            is_open=False,
        ),

    ],
    id="tldr"
)


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
