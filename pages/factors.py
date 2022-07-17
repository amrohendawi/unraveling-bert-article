from dash import html

from utils import textBox

import pages.subsections.layer_epoch_effect as layer_epoch_effect
import pages.subsections.fine_tuning as fine_tuning
import pages.subsections.task_relevance as task_relevance
import pages.subsections.dataset as dataset
import pages.subsections.model_size as model_size
import pages.subsections.model_depth as model_depth

# Text sections
factors_section = html.Div([
    textBox(
        """
        ## What makes a model more transferable? 
        In this section we discuss the main factors that affect the
        transferability of BERT. These factors are the number of layers,
        the number of fine-tuning epochs, the dataset size, and the label
        noise.
        """,
        text_id="factors"
    ),
],
)

layout = html.Div(
    [
        factors_section,
        dataset.content,
        model_size.content,
        model_depth.content,
        layer_epoch_effect.content,
        task_relevance.content,
        fine_tuning.content,
    ],
    className="row",
)
