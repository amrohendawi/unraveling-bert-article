import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from utils import textBox, DATA_PATH, df_to_matrix
from appServer import app


content = html.Div([
    textBox(
        """
            #### How big should BERT be?
            
            The size of BERT depends on the specific application and dataset.
            Generally, BERT should have 30-40% of weights pruned without any impact on downstream tasks.
            However, too many BERT heads and layers can be harmful to the performance of downstream tasks.
            Therefore, it is often best to train a larger model and then compress it.
            
            The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks.
            Additionally, compression can make BERT more transferable.
            """,
        text_id="model-size"
    ),
])