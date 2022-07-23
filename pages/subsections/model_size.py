import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from utils import textBox, DATA_PATH, df_to_matrix
from appServer import app


content = html.Div([
    html.H3("How big should BERT be?"),
    html.P("The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
    html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
           html.A("[5]", id="t5-ref", href="#references")]),
    html.P(["The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
           html.A("[6", id="t6-ref", href="#references")]),  # TODO: elaborate a little on the texts here
    html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
           html.A("[7]", id="t7-ref", href="#references")]),
    html.P(["The following tabel shows many version of Tranformer with many options of comparsion and the related speed up. ",
           html.A("[8]", id="t8-ref", href="#references")]),
    html.Div(html.Img(src=app.get_asset_url('bert-size.png'),
             className="ds_img"), className="img_holder"),
    html.P(" It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    html.Hr()
])
