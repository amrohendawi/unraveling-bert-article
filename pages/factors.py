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
    html.H3("What makes a model more transferable?"),
    html.Div(html.Img(src="http://i1.sndcdn.com/artworks-000018461619-qq72il-original.jpg?77cede1", className="img_bert"),
             title="http://i1.sndcdn.com/artworks-000018461619-qq72il-original.jpg?77cede1", className="img_holder_bert"),
    html.P("We will first discuss the effect of the dataset used in the source and target task. The next section will discuss the size of BERT and its impact on the results. Then, the depth of the model and its relation to transfer learning will be explored."),
    html.P("In the section \"Layer and epoch effect on transferability\", we will try to visualise the layer of the model against some epochs in multi setups."),
    html.P("After that, the task task relevance will be highlighted with a heat-map. By the end, the techniques used during the fine tuning and its effect on the results will be considered."),
    html.Hr()
],
    id="factors"
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
