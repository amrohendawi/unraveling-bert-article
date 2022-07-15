import dash_bootstrap_components as dbc
from dash import html
from utils import textBox, text_article

introduction_text_dict = [
    {"type": "txt",
     "text": """
            ## Introduction
                Since their introduction in 2017, Transformers (Vaswani et al.,
                2017) have taken NLP by storm, offering enhanced parallelization and
                better 
            """
     },
    {"type": "popup",
     "text": """modeling""",
     "hovtxt": "the action or habit of estimating something as"},
    {"type": "txt",
     "text": """
                 comtinue their introduction in 2017, Transformers (Vaswani et al.,
                2017) have taken NLP by storm, offering enhanced parallelization and
                better
            """,
     },
    {"type": "ref",
     "text": """[[12]](/#references)""",
     "hovtxt": """
    >
    > Block quotes are used to highlight text with the link[www.google.com](www.google.com).
    >
    """},
    {"type": "txt",
     "text": """
                 comtinue their introduction in 2017, Transformers (Vaswani et al.,
                2017) have taken NLP by storm, offering enhanced parallelization and
                better
            """,
     },
    {"type": "popup",
     "text": """someHOVERABLETEXT""",
     "hovtxt": "HOVERHOVERHOVER"},
]

layout = html.Div(
    [
        text_article(introduction_text_dict,  style=""),
        textBox(
            """
             of long-range dependencies. The best known
                Transformer-based model is BERT (Devlin et al., 2019); it obtained
                state-of-the-art results in numerous benchmarks and is still a
                must-have baseline.

                BERT is a deep learning model that has been shown to be effective
                for a range of natural language understanding tasks. However, it is
                not clear how well BERT transfers to more specific tasks. In this
                article, we investigate the factors that affect the transferability
                of BERT. These factors include the number of layers, the number of
                fine-tuning epochs, the dataset size, and the label noise.

                We first consider the linguistic aspects of it, such as the number
                of layers, the number of fine-tuning epochs, and the dataset size.
                We find that, in general, a shallower BERT model transfers better
                than a deeper model, and that a model trained with more data and for
                more epochs transfers better than a model trained with less data and
                for fewer epochs. We also consider the effect of label noise on
                transferability. We find that label noise has a negative effect on
                transferability.

                In conclusion, we find that the transferability of BERT is affected
                by a number of factors, including the number of layers, the number
                of fine-tuning epochs, the dataset size, and the label noise.
            """
        ),
        html.A('[1]', href='#references', id="refAnchor"),
    ],
    id="introduction",
    className="row",
)
