import dash_bootstrap_components as dbc
from dash import html, Output, Input, callback, dcc

layout = html.Div(
    [
        html.Div(
            [
                html.H3("TL;DR"),
                html.Br(),
                html.P(
                    """
                    This work evaluates the effectiveness of transfer learning on BERT (generic language model) and
                    investigates the factors and trade-offs that influence the acquisition of a new downstream task
                    based on the latest scientific research.
                    """,
                ),
                html.P(
                    """
                    The study found that the model's performance is highly influenced by the size of the training data,
                    the similarity between the tasks, the amount of fine-tuning, and the optimization method used.
                    In terms of layer depth, the study found that the knowledge contained in the middle layers is the most transferable and vital.
                    """
                ),
            ], id="tldr"
        ),
        html.Hr(),
        # TODO: Put the following in a component. Maybe instad of direction compnent (I think it is not necessary)
        dbc.Offcanvas(
            html.Ul([
                dbc.Tooltip(
                    html.A(
                        "Lundberg, Scott M., and Su-In Lee. \"A unified approach to interpreting model predictions.\" Advances in neural information processing systems 30 (2017).",
                        href="https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
                        target="_blank"),
                    target="ref-1",
                    delay={"show": 0, "hide": 1000},
                    placement='left',
                    class_name="custom_tooltip",
                ),
                html.Li(
                    "The SHAP (SHapley Additive exPlanations) framework is a tool for explaining the output of machine learning models."),
                html.Li(
                    "It is based on the concept of Shapley values from game theory, and aims to provide explanations that are fair, consistent, and locally accurate."),
                html.Li(
                    "SHAP values can be used to measure the importance of each feature to the model output, and can be used to explain individual predictions.", ),
                html.Li(
                    "The SHAP framework has been implemented in a number of popular machine learning libraries, including XGBoost, LightGBM, and CatBoost.", ),
                html.Li([
                    "When interpreting SHAP values, it is important to bear in mind that they represent the marginal contribution of a feature to the model output, rather than the absolute contribution. This means that the SHAP values for a given feature can vary depending on the values of other features. ",
                    html.P("[1]", id="ref-1", className="ref-link"), ])]),
            id="toggle-shap-offcanvas",
            title="SHAP",
            is_open=False,
            placement="end"
        ),
        html.Div([
            html.H3("Fine-Tuning Effect Demo"),
            html.P([
                "This demo showcases an explainable framework for sentiment analysis to demonstrate the effect of fine-tuning. The frame work ",
                html.A(
                    "SHAP",
                    id="toggle-shap",
                    className="toggle-text",
                ),
                " visualizes the importance of each token and its attribution to the overall result. In the current set up, we would like to show the effect of fine-tuning using a base model and another model that is fine tuned to detect sentiment."]),
            dcc.Dropdown(
                ['love, what for a beautiful weather today', 
                'the picture was ugly, in addition the frame was bad', 
                'the film was very cool and the players were perfect',
                'the tree is very good but the nest is very bad',
                'fear leads to anger, anger leads to hate, hate leads to suffering',
                ],
                value = "love, what for a beautiful weather today",
                searchable=False,
                clearable=False,
                id = "demo-dropdown"
            ),
            html.Div(id='demo-container', children=''),
            html.Hr()

        ], className="card-component")
    ],
)



@callback(
    Output("toggle-shap-offcanvas", "is_open"),
    Input("toggle-shap", "n_clicks"),
)
def toggle_text(n1):
    return n1
