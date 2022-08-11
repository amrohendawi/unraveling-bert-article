import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from pages.references import references_dict
from utils import add_tooltip
from appServer import app

text_content = html.Div(
    [
        html.H4("Does depth matter?"),
        html.Br(),
        html.P(
            """
            Although the transferability of BERT models appears to be depth-agnostic, i.e. shallower or deeper BERT
            models can be transferred to new tasks with comparable performance, research has found that the layers play
            different roles in encoding information in the model.
            """),
    ],
    id="model-depth"
)

content = html.Div([
    text_content,
    html.P(
        [
            "The next visualizations help in illustrating the difference between the layers of different langauge models ",
            add_tooltip(references_dict[9]['title'], "10", "ref-10-1", href=references_dict[9]['href']),
            "."
        ]),
    html.Div([
        dcc.Dropdown(['ELMo (original)', 'ELMo (4-layer)', 'ELMo (transformer)', 'OpenAI transformer',
                      'BERT (base, cased)', 'BERT (large, cased)'], 'BERT (base, cased)', id='depth-dropdown',
                     clearable=False),
        html.Div(html.Img(src=app.get_asset_url('depth/bertb.png'),
                          className="depth_img", id="depth_img"), className="depth_img_holder"),
        html.Div(html.Img(src=app.get_asset_url('depth/footer.png'),
                          className="depth_footer"), className="depth_footer_holder"),
    ]),
    html.P("Based on the study, the following conclusions were drawn:"),
    html.Ul([
        html.Li(
            [
                "The lower layers are more general and can be applied to a wider range of tasks ",
                add_tooltip(references_dict[10]['title'], "11", "ref-11-1", href=references_dict[10]['href']),
                "."
            ]
        ),
        html.Li(
            [
                "The middle layers are more transferable and can be used for other tasks ",
                add_tooltip(references_dict[9]['title'], "10", "ref-10-2", href=references_dict[9]['href']),
                "."
            ]
        ),
        html.Li(
            [
                "The final layers are the most task-specific ",
                add_tooltip(references_dict[9]['title'], "10", "ref-10-3", href=references_dict[9]['href']),
                "."
            ]
        ),
    ]),
    html.P(
        """
        Understanding the differences between the layers of different language models can be helpful in other factors
        that affect the transferability of models.
        For example, depending on the task, knowing which weights in which layers to freeze during fine-tuning can
        have a significant impact on the final performance of the model.
        """
    ),
    html.Hr()
],
)


@app.callback(
    Output('depth_img', 'src'),
    Input('depth-dropdown', 'value')
)
def update_output(value):
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
    # return f'You have selected {value}'
