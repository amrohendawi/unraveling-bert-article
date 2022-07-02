from dash import html
from utils import textBox

layout = html.Div(
    [
        textBox(
            """
    ## Conclusion
    Fine-tuning can improve a model's transferability, although the extent of the improvement may vary depending on the task and domain.
    Other factors like the dataset and the model's size  can also affect transferability.
    """),
    ],
    id="conclusion",
    className="row",
)
