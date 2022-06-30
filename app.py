# -*- coding: utf-8 -*-
# Import required libraries
from dash import dcc, html, Input, Output, callback_context, State
import dash_bootstrap_components as dbc
from appServer import app
from dash.exceptions import PreventUpdate

from pages import tldr, introduction, factors, directions, conclusion, references
from utils import textBox

server = app.server

# dictionary for the headlines from TEXTS
HEADLINES = {
    'tldr': {"index": 0, "title": "TL;DR", "type": "paragraph"},
    'introduction': {"index": 10, "title": "Introduction", "type": "paragraph"},
    'factors': {"index": 20, "title": "Factors", "type": "section"},
    'fine-tuning': {"index": 27, "title": "Fine-tuning", "type": "subsection"},
    'layer-epoch': {"index": 31, "title": "Layer & Epoch", "type": "subsection"},
    'dataset': {"index": 35, "title": "Dataset", "type": "subsection"},
    'task-similarity': {"index": 39, "title": "Task Similarity", "type": "subsection"},
    'embeddings-quality': {"index": 43, "title": "Embeddings Quality", "type": "subsection"},
    'directions': {"index": 50, "title": "Directions & Further Research", "type": "paragraph"},
    'conclusion': {"index": 60, "title": "Conclusion", "type": "paragraph"},
    'references': {"index": 70, "title": "References", "type": "paragraph"},
}

headlines_style = {
    "paragraph": "nav-link-paragraph",
    "section": "nav-link-section",
    "subsection": "nav-link-subsection",
}

navLinks = []
for key in HEADLINES:
    navLinks.append(
        dbc.NavLink(
            HEADLINES[key]["title"],
            href="#" + key,
            id=key + "-button",
            className=headlines_style[HEADLINES[key]["type"]],
            external_link=True,
        )
    )

sidebar = html.Div(
    [
        html.H3("Content", className="display-4",
                style={"font-style": "bold", "padding": "0rem 1rem 0rem 1rem"}),
        html.Hr(style={"margin": "0rem 1rem 0rem 1rem"}),
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Slider(
                                min=0,
                                max=max([HEADLINES[key]['index']
                                         for key in HEADLINES]),
                                # read value from function called read_value
                                value=0,
                                step=1,
                                marks={i: {'style': {
                                    'transform': 'scaleY(-1)',
                                    }} for i in [HEADLINES[key]['index'] for key in HEADLINES]},
                                id="slider-vertical",
                                vertical=True,
                            ),
                            style={'transform': 'scaleY(-1)',
                                   'flex': '0 0 0%',
                                   }
                        ),
                        dbc.Col(
                            dbc.Nav(
                                navLinks,
                                vertical=True,
                            ),
                        )
                    ], style={"padding-left": "0px"}
                ),
            ],
            style={"padding": "1rem"},
        )
    ],
    className="sidebar card-component",
    style={
        "top": "20%"
    },
)

body = html.Div([
    html.Div(
        [
            textBox(
                """
                    # Unraveling BERT's Transferability
                    """,
                className="title",
                style={"textAlign": "center", "width": "fit-content"}
            ),
        ],
        className="row",
        style={"marginTop": "25px"},
    ),
    tldr.layout,
    introduction.layout,
    factors.layout,
    directions.layout,
    conclusion.layout,
    references.layout,

], style={
    "padding": "2rem",
})

################### App layout ###################
app.layout = html.Div(
    dbc.Row(
        [
            dcc.Location(id="url"),
            dcc.Store(id="click-output"),
            dbc.Col(sidebar, width=2),
            dbc.Col(body, width=9),
        ],
        style={
            "justify-content": "center",
        },

    )
)

buttons_list = [k + '-button' for k in HEADLINES]


@app.callback(Output('slider-vertical', 'value'),
              [Input(button, 'n_clicks')
               for button in buttons_list],
              )
def update_value(n_clicks, *args):
    ctx = callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id in buttons_list:
            return HEADLINES[prop_id[:-7]]['index']


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
    # app.run_server(debug=False)
