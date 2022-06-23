from dash import Dash
import dash_bootstrap_components as dbc

# Setup the app
app = Dash(
    __name__, meta_tags=[
        {"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
app.title = "Unraveling BERT's transferability"

app.config.suppress_callback_exceptions = True
