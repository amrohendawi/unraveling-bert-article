from dash import html, dcc
from utils import textBox
import dash_bootstrap_components as dbc
from dash import html, Output, Input, callback

layout = html.Div(
    [
        html.Div(
            [
                html.H3("TL;DR"),
                html.Br(),
                html.P(
                    """
                    In this work, we investigate the extent to which a BERT model trained on general natural language understanding
                    can be transferred to downstream, more specific tasks. We find that the model is able to transfer its knowledge
                    to new tasks to some extent, but that the amount of knowledge transfer varies depending on the similarity between
                    the training and test tasks.
                    """,
                ),
                html.P(
                    """
                    For example, the model is able to transfer more knowledge to a task that is similar to the task it was trained on than to a task that is dissimilar. This suggests that the model is able to learn general principles that can be applied to new tasks, but that the amount of knowledge that can be transferred is limited by the similarity between the tasks.
                    """
                ),
                html.P(
                    """
                    Furthermore, the results show that the effectiveness of the system increases with the number of layers, number of fine-tuning epochs, and dataset size used for the fine-tuning. In addition, a shallower BERT model transfers better than a deeper model.
                    """
                ),
            ], id="tldr"
        ),
        html.Hr(),
        # TODO: Put the following in a component. Maybe instad of direction compnent (I think it is not necessary)
        dbc.Offcanvas(
            html.Ul([
                dbc.Tooltip(
                    html.A("Lundberg, Scott M., and Su-In Lee. \"A unified approach to interpreting model predictions.\" Advances in neural information processing systems 30 (2017).",
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
            html.P("The demo can take some minutes to display the content"),
            html.Div([dbc.Input(id="input-on-submit", placeholder="Type a sentence here...", type="text"),
                      dbc.Button("Submit", color="info", className="me-1", id='submit-val', n_clicks=0)]),
            html.Div(id='container-button-basic', children=''),
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
