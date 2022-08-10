from dash import html, dcc

layout = html.Div(
    [
        dcc.Markdown(
            """
    ### Conclusion
    
    Fine-tuning can improve a model's transferability, although the extent of the improvement may vary depending on the task and domain.
    Other factors like the dataset and the model's size  can also affect transferability.
    """),
    ],
    id="conclusion",
    className="row",
)
