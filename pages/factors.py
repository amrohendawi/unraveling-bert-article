from dash import html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import pages.subsections.layer_epoch_effect as layer_epoch_effect
import pages.subsections.fine_tuning as fine_tuning
import pages.subsections.task_relevance as task_relevance
import pages.subsections.dataset as dataset
import pages.subsections.model_size as model_size
import pages.subsections.model_depth as model_depth

# Text sections
factors_section = html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H3("What makes a model more transferable?"),
                    html.Br(),
                    html.P(
                        """
                        In this section we will discuss some of the factors that make a model more transferable.
                        These factors are divided into three categories:
                        """
                    ),
                    html.Ol([
                        html.Li("Related to the model's architecture, and how well it can be adapted to different tasks."),
                        html.Li("Related to the data used to train the model, and how well it can be generalized to new data."),
                        html.Li("Related to the model training, including optimization methods and taskonomy."),
                    ]),
                ],
            ),
            dbc.Col(
                dmc.Image(
                    src="http://i1.sndcdn.com/artworks-000018461619-qq72il-original.jpg?77cede1",
                    fit="contain",
                    style={"margin": "1rem 0"},
                ),
                width=3,
            ),
        ],
    ),
    html.Hr()
],
    id="factors"
)

layout = html.Div(
    [
        factors_section,
        dataset.content,
        model_size.content,
        model_depth.content,
        layer_epoch_effect.content,
        task_relevance.content,
        fine_tuning.content,
    ],
    className="row",
)
