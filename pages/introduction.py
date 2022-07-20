from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        dbc.Tooltip(
            "Vu, T., Wang, T., Munkhdalai, T., Sordoni, A., Trischler, A., Mattarella-Micke, A., Maji, S. and Iyyer, M., 2020. Exploring and predicting transferability across NLP tasks. arXiv preprint arXiv:2005.00770.",
            target="tt-1",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Scrolling modal")),
                dbc.ModalBody("LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM LOREM "),
            ],
            id="modal-scroll",
            is_open=False,
        ),
        dbc.Offcanvas(
            html.P("Some offcanvas content..."),
            id="offcanvas-placement",
            title="Placement",
            is_open=False,
            placement="end"
        ),
        html.H3("BERT and Transfer-Learning"),
        html.P(
            [
                "Since their introduction in 2017, Transformers (Vaswani et al., 2017) have taken NLP by storm, offering enhanced parallelization and better modeling of long-range ",
                html.A("dependencies.", id="tt-1", href="#references"),
            ]
        ),
        html.P(
            [
                "The best known Transformer-based model is BERT(Devlin et al., 2019); it obtained state-of-the-art results in numerous benchmarks and is still a must-have baseline.",
            ]
        ),
        html.P(
            "BERT is a deep learning model that has been shown to be effective \
            for a range of natural language understanding tasks.However, it is \
            not clear how well BERT transfers to more specific tasks.In this \
            article, we investigate the factors that affect the transferability \
            of BERT.These factors include the number of layers, the number of \
            fine-tuning epochs, the dataset size, and the label noise."
        ),
        html.P(
            "We first consider the linguistic aspects of it, such as the number \
            of layers, the number of fine-tuning epochs, and the dataset size. \
            We find that, in general, a shallower BERT model transfers better \
            than a deeper model, and that a model trained with more data and for \
            more epochs transfers better than a model trained with less data and \
            for fewer epochs.We also consider the effect of label noise on \
            transferability.We find that label noise has a negative effect on \
            transferability."
        ),
        html.P(
            [
                "In conclusion, we find that the transferability of BERT is affected by a number of factors, including the number of layers, the number of fine - tuning epochs, the dataset size, and the label noise.",
                html.A(
                    "Click to toggle the modal",
                    id="open-scroll",
                ),
                " or ",
                html.A(
                    "Click to toggle the offCanvas",
                    id="open-scroll2",
                    className="toggle-text",
                ),
            ]
        ),
    ],
    className="text-box card-component row",
    id="introduction",
)


@callback(
    Output("modal-scroll", "is_open"),
    Input("open-scroll", "n_clicks"),
)
def toggle_text(n1):
    return n1


@callback(
    Output("offcanvas-placement", "is_open"),
    Input("open-scroll2", "n_clicks"),
)
def toggle_text(n1):
    return n1
