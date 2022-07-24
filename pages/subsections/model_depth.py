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
    html.Div([
    dcc.Dropdown(['ELMo (original)', 'ELMo (4-layer)', 'ELMo (transformer)', 'OpenAI transformer', 'BERT (base, cased)', 'BERT (large, cased)'], 'BERT (base, cased)', id='depth-dropdown', clearable=False),
    html.Div(html.Img(src=app.get_asset_url('depth/bertb.png'), className="depth_img", id="depth_img"), className="depth_img_holder"),
    html.Div(html.Img(src=app.get_asset_url('depth/footer.png'),className="depth_footer"), className="depth_footer_holder"),
]),
    #html.Div(html.Img(src=app.get_asset_url('depth/bertb.png'), className="depth_img"), className="img_holder"),
    #html.Div(html.Img(src=app.get_asset_url('depth/footer.png'),className="depth_img"), className="img_holder"),
    # TODO: write more here
    html.P("We could conclude that the middle layer are the most caplabe to transfer knowledge"),
    html.Hr()
],
)

@app.callback(
    Output('depth_img', 'src'),
    Input('depth-dropdown', 'value')
)
def update_output(value):
    print(value)
    if value == "ELMo (original)":
        return app.get_asset_url('depth/elmo.png')
    elif value == "ELMo (4-layer)":
        return app.get_asset_url('depth/elmo4.png')
    elif value == "ELMo (transformer)":
        return app.get_asset_url('depth/elmot.png')
    elif value == "OpenAI transformer":
        return app.get_asset_url('depth/openait.png')
    elif value == "BERT (base, cased)":
        return app.get_asset_url('depth/bertb.png')
    elif value == "BERT (large, cased)":
        return app.get_asset_url('depth/bertl.png')
    else:
        return app.get_asset_url('depth/bertb.png')
    #return f'You have selected {value}'
