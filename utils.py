from dash import dcc, html
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

def textBox(text, style=None, className=""):
    return html.Div(dcc.Markdown(
        text.replace(
            "  ", ""
        ),
    ),
        className="text-box card-component " + className, style=style,
    )
