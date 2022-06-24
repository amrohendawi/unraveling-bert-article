from dash import dcc, html

layout = html.Div(
    [
        html.Div([dcc.Markdown(
            """
    # Conclusion
    Fine-tuning can improve a model's transferability, although the extent of the improvement may vary depending on the task and domain.
    Other factors like the dataset and the model's size  can also affect transferability.
    """.replace(
                "  ", ""
            ),
        )],
            className="text-box card-component"),
    ],
    id="page",
    className="row",
)
