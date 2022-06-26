# -*- coding: utf-8 -*-
# Import required libraries
from dash.dependencies import Input, Output
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
# import dash_cytoscape as cyto
from appServer import app

from pages import tldr, introduction, factors, directions, conclusion, references

server = app.server

# add some padding.
CONTENT_STYLE = {
    "margin-left": "10rem",
    "padding": "2rem",
}

# dictionary for the headlines from TEXTS
HEADLINES = {
    0: "TL;DR",
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

sidebar = html.Div(
        [
            html.H3("Content", className="display-4", style= {"font-style": "bold", "padding": "0rem 1rem 0rem 1rem"}),
            html.Hr(style={"margin": "0rem 1rem 0rem 1rem"}),
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
                                    'transform': 'scaleY(-1)', 'fontSize': '12px' if i % 10 == 0 else '10',
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
        className="sidebar card-component",
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
                className="title card-component",
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
            dbc.Col(sidebar, width=3, style={
                "position": "fixed",
                "z-index": "1",
                "top": "0",
                "overflow-x": "hidden",
                }),
            dbc.Col(body, width=9),
        ],
        style={
            "justify-content":"center",
        },
        
    )
)

@app.callback(Output('page-content', 'children'),
              [Input("slider-vertical", "value")])
def display_page(value):
    if value == 0:
        return tldr.layout
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
