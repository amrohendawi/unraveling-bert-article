from dash import html, dcc

layout = html.Div(
    [
        dcc.Markdown(
            """
    ### Conclusion
    
    In this work we have investigated a number of factors affecting the transferability of language models and, in
    particular, BERT models. Some factors considered are dataset size, model size, model layer depth, task relevance,
    layer and epoch effect, and some fine-tuning techniques.
    
    We found that the size of the training data is a very important factor for the success of transfer learning.
    The model's performance is also highly influenced by the similarity between the tasks, the amount of fine-tuning,
    and the optimization method used. In terms of layer depth, the study found that the knowledge contained in the
    middle layers is the most transferable and vital.
    
    There is still much research that needs to be done in order to fully understand what and how different factors
    influence BERT's transferability. There is no one-size-fits-all answer to  the optimal settings for transfer learning.
    The right settings depend on the specifics of the task and the data. In your next BERT project,  consider these
    findings and experiment with different settings to find what works best for you.
    """),
    ],
    id="conclusion",
    className="row",
)
