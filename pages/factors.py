from dash import html

from utils import textBox

import pages.subsections.layer_epoch_effect as layer_epoch_effect
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

fine_tuning_section = html.Div([
    textBox(
        """
            ## Fine-tuning
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.

            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """
    ),
], id="fine-tuning"
)

layout = html.Div(
    [
        factors_section,
        fine_tuning_section,
        layer_epoch_effect.content,
        dataset.content,
        task_similarity.content,
        embeddings_quality.content,
    ],
    id="factors",
    className="row",
)
