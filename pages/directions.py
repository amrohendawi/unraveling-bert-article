from dash import html, dcc

layout = html.Div(
    [
        dcc.Markdown(
            """
        ### Directions and Further Research
        
        
        There is still much research that needs to be done in order to fully understand what and how different factors 
        BERT's transferability. We list some of the open questions for further research:
        
        - Is there a specific number of fine-tuning samples or a specific amount of fine-tuning that is optimal for transfer-learning?
        - Is it possible to identify which language model layers are most responsible for the transfer of knowledge?
        - Is there an ideal model size for transfer learning?
        - Are there certain languages that are easier to transfer between than others?
        
        Answers to these questions will help researchers to better understand how BERT works and how to get the most out of it.
        """
        ),
    ],
    id="directions",
    className="row",
)
