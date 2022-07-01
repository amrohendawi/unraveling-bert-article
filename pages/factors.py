from dash import html

from utils import textBox

import pages.subsections.layer_epoch_effect as layer_epoch_effect
import pages.subsections.fine_tuning as fine_tuning
import pages.subsections.task_similarity as task_similarity
import pages.subsections.embeddings_quality as embeddings_quality
import pages.subsections.dataset as dataset

# Text sections
factors_section = html.Div([
    textBox(
        """
        # Factors
        In this section we discuss the main factors that affect the
        transferability of BERT. These factors are the number of layers,
        the number of fine-tuning epochs, the dataset size, and the label
        noise.
        """
    ),
])


layout = html.Div(
    [
        factors_section,
        fine_tuning.content,
        layer_epoch_effect.content,
        dataset.content,
        task_similarity.content,
        embeddings_quality.content,
    ],
    id="factors",
    className="row",
)
