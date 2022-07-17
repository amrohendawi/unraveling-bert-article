from dash import dcc, html, Input, Output

from utils import textBox, DATA_PATH


content = html.Div([
    textBox(
        """
            #### Embeddings Quality
            The quality of the learned embeddings is an important factor in the performance of downstream tasks. If the embeddings are of poor quality, the downstream task will likely suffer.
            There are a few ways to measure the quality of learned embeddings. One is to evaluate the performance of a model that is trained on a supervised task using the learned embeddings as features. Another is to evaluate the performance of a model that is trained on a unsupervised task using the learned embeddings as features.
            BERT models have been shown to produce high-quality embeddings. For example, a study found that a BERT model trained on a large corpus of English text produced embeddings that were better at capturing syntactic and semantic information than word2vec embeddings.
            """,
        text_id="embeddings-quality"
    ),
]
)
