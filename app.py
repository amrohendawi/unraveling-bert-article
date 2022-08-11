# -*- coding: utf-8 -*-
# Import required libraries
from dash import dcc, html, Input, Output, callback_context, State
import dash_bootstrap_components as dbc
from appServer import app
from dash.exceptions import PreventUpdate
import dash_dangerously_set_inner_html
from pages import tldr, introduction, factors, directions, conclusion, references
import requests

server = app.server

# dictionary for the headlines from TEXTS
HEADLINES = {
    'tldr': {"index": 0, "title": "TL;DR", "type": "paragraph"},
    'introduction': {"index": 10, "title": "BERT and Transfer-Learning", "type": "paragraph"},
    'factors': {"index": 20, "title": "What makes a model more transferable?", "type": "section"},
    'dataset': {"index": 27, "title": "The Effect of Dataset Size", "type": "subsection"},
    'model-size': {"index": 31, "title": "How big should BERT be?", "type": "subsection"},
    'model-depth': {"index": 35, "title": "Does depth matter?", "type": "subsection"},
    'layer-epoch': {"index": 39, "title": "Layer & Epoch", "type": "subsection"},
    'task-relevance': {"index": 43, "title": "The Importance of Task Relevance", "type": "subsection"},
    'fine-tuning': {"index": 47, "title": "Fine-tuning", "type": "subsection"},
    'directions': {"index": 55, "title": "Directions & Further Research", "type": "paragraph"},
    'conclusion': {"index": 65, "title": "Conclusion", "type": "paragraph"},
    'references': {"index": 75, "title": "References", "type": "paragraph"},
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
                style={"fontStyle": "bold", "padding": "0rem 1rem 0rem 1rem"}),
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
                                   'paddingRight': '0rem',
                                   }
                        ),
                        dbc.Col(
                            dbc.Nav(
                                navLinks,
                                vertical=True,
                            ),
                            style={
                                'paddingLeft': '0rem',
                            }
                        )
                    ], style={"paddingLeft": "0px"}
                ),
            ],
        )
    ],
    className="sidebar",
)

body = html.Div([
    html.Div(
        [
            html.H1("Unraveling BERT's Transferability",
                    className="title text-box card-component",
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
            dbc.Col(sidebar, lg=3),
            dbc.Col(body, lg=7, width=12, style={"border-left": "1px solid black"}),
        ],
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


@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value')

)
def update_output(n_clicks, value):
    if value != None and n_clicks == 1:
        url = "http://kadi.dnsfor.me:5020"
        response = requests.post(url, json={'sentence': value})
        return html.Div([
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(response.content.decode("utf-8").split("$")[0]),
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(response.content.decode("utf-8").split("$")[1]),
            html.P(
                "When we compare the results of the fine-tuned model with the original model, we can see that some of the token values and importance have been adjusted.")
        ])


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
