from dash import html, dcc

layout = html.Div(
    [
        dcc.Markdown(
            """
    ### Directions and Further Research
    There is still much research that needs to be done in order to fully understand how fine-tuning affects BERT's transferability.
    In particular, it would be interesting to compare the transferability of models that are fine-tuned on different tasks and domains.
    Additionally, it would be helpful to investigate how different amounts of fine-tuning affect transferability.
    """
        ),
    ],
    id="directions",
    className="row",
)
