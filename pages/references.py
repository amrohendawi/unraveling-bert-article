from dash import dcc, html

layout = html.Div(
    [
        html.Div([dcc.Markdown(
            """
    # References
    1. https://arxiv.org/pdf/1906.01083.pdf
    2. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
    3. https://huggingface.co/transformers/pretrained_models.html#bert
    4. https://medium.com/@jonathan_hui/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
    """.replace(
                "  ", ""
            ),
        )],
            className="text-box"),
    ],
    id="page",
    className="row",
)
