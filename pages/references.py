from dash import html, dcc
from utils import textBox
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        dcc.Markdown(
            """
            ## References
            """
            , id="references"
        ),
        # a list of bullet points
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    "1. Vu, Tu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. \"Exploring and predicting transferability across NLP tasks.\" arXiv preprint arXiv:2005.00770 (2020).",
                    href="https://arxiv.org/pdf/1906.01083.pdf",
                ),
                dbc.ListGroupItem(
                    "2. Vu, Tu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. \"Exploring and predicting transferability across NLP tasks.\" arXiv preprint arXiv:2005.00770 (2020).",
                    href="https://arxiv.org/pdf/1906.01083.pdf",
                ),
                dbc.ListGroupItem(
                    "3. Vu, Tu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. \"Exploring and predicting transferability across NLP tasks.\" arXiv preprint arXiv:2005.00770 (2020).",
                    href="https://arxiv.org/pdf/1906.01083.pdf",
                ),
            ]
        ),
    ],
    className="row text-box card-component ",
)
