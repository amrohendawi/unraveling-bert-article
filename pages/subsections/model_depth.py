import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from utils import textBox, DATA_PATH, df_to_matrix
from appServer import app

text_content = html.Div(
    [
        html.H4("Does depth matter?"),
        html.P(
            """
            The transferability of the bert model is affected by the depth of the model. The shallower the model,
            the more transferable it is.
            """),
    ],
    id="model-depth"
)

content = html.Div([
    text_content,
    html.P(
        [
            "The next plot shows BERTs with many numbers of layrs being analyzid to detect the most transferabel layers",
            html.A("[9].", id="t9-ref", href="#references")
        ]),
    html.Div(html.Img(src=app.get_asset_url('layer-performance.png'),
                      className="depth_img"), className="img_holder"),
    # TODO: write more here
    html.P("We could conclude that the middle layer are the most caplabe to transfer knowledge"),
    html.Hr()
],
)
