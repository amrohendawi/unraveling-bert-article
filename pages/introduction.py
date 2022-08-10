from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from utils import add_tooltip
from pages.references import references_dict

offCanvases = [
    dbc.Offcanvas(
        html.Ul([
            dbc.Tooltip(
                html.A(
                    references_dict[1]['title'],
                    href=references_dict[1]['href'],
                ),
                target="ref-2-offcanvas",
                delay={"show": 0, "hide": 1000},
                placement='left',
                class_name="custom_tooltip",
            ),
            html.Li(
                "Bert is a transformer-based machine learning algorithm that was designed to pre-train deep bidirectional representations from natural language text by masking a percentage of the input tokens and training the network to predict the masked tokens."),
            html.Li(
                "BERT has been shown to be very successful at a wide range of natural language processing tasks such as question answering, text classification, and language modeling."),
            html.Li(
                "BERT comes in two version. The first one is the base model, which contains 12 layer and the large one, which has 24 layer "),
            html.Li(
                "The BERT architecture is based on the Transformer architecture, which is a self-attention based model. ", ),
            html.Li(
                "BERT is trained using two training objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).", ),
            html.Li(
                "In MLM, a percentage of the input tokens are randomly masked and the model is trained to predict the masked tokens.", ),
            html.Li(
                "In NSP, the model is trained to predict whether two sentences are consecutive or not.", ),
            html.Li(
                "The model can be fine-tuned for specific tasks by adding a task-specific layer on top of the pre-trained model. For example, to fine-tune Bert for text classification, a task-specific classification layer can be added.", ),
            html.Li([
                "BERT is a very powerful model that has achieved state-of-the-art results on a wide range of natural language processing tasks.",
                html.P("[2]", id="ref-2-offcanvas", className="ref-link")
            ])]),
        id="toggle-bert-offcanvas",
        title="BERT",
        is_open=False,
        placement="end"
    ),
    dbc.Offcanvas(
        html.Ul([
            dbc.Tooltip(
                html.A(
                    "S. J. Pan and Q. Yang, \"A Survey on Transfer Learning,\" in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010, doi: 10.1109/TKDE.2009.191.",
                    href="https://ieeexplore.ieee.org/abstract/document/5288526",
                    target="_blank"),
                target="ref-3-offcanvas",
                delay={"show": 0, "hide": 1000},
                placement='left',
                class_name="custom_tooltip",
            ),
            html.Li(
                "Transfer Learning is a technique that can be used to improve the performance of a machine learning model by using knowledge from a related task. The main idea is to transfer knowledge from a model that has been trained on a large and high-quality dataset to a model that is trained on a smaller and lower-quality dataset."),
            html.Li(
                "The main advantage of transfer learning is that it can be used to train a model on a dataset that is too small or too low quality to train a model from scratch. Additionally, transfer learning can be used to improve the performance of a machine learning model by using knowledge from a related task with less data and computional power."),
            html.Li([
                "The main disadvantage of transfer learning is that it is difficult to control the amount of knowledge that is transferred from the source model to the target model. Additionally, transfer learning can only be used to improve the performance of a machine learning model if the source and target datasets are similar.",
                html.P("[3]", id="ref-3-offcanvas", className="ref-link"),
                " One way to avoid this drawback is to freeze some of the weights. We will highlight this technique more in the next sections.",
            ])]),
        id="toggle-tl-offcanvas",
        title="Transfer Learning",
        is_open=False,
        placement="end"
    ),
]

text_content = html.Div(
    [
        html.H3("BERT and Transfer-Learning"),
        html.Br(),
        html.Div(html.Img(src="assets/transfer-learning.png",
                          className="img_tl",
                          title="https://editor.analyticsvidhya.com/uploads/444861_2vse-G3FKMT2D59NFWduMA.jpg"),
                 className="img_holder"),
        html.P(["""
                Since its release in late 2018, The Bidirectional Encoder Representations from Transformers 
                """,
                add_tooltip("Learn more about BERT by clicking here.", "BERT", "toggle-bert"),
                """
                is a state-of-the-art natural language processing (NLP) technique employed for training self-supervised learning language models.
                Moreover, BERT revolutionized how machines understand human language by following a bidirectional transformer design that learns deep
                bidirectional representations from an unlabeled corpus by jointly conditioning both left and right contexts in all layers. This learning
                process is achieved by using the simple tasks of Masked Language Modeling and Next Sentence Prediction. Finally, BERT generalized language
                understanding allows it to be extended to new domains such as text classification, question answering, and text generation.
                """,
                # TODO: customize max width of reference text
                #  maybe we can define a general style for all references
                add_tooltip(references_dict[1]['title'],
                            "2",
                            "ref-2",
                            href=references_dict[1]['href'],
                            ),
                ]),
    ],
    id="introduction"
)

layout = html.Div(
    [
        *offCanvases,
        text_content,
        html.P([
            add_tooltip("Click to learn more about transfer-learning.", "Transfer learning", "toggle-tl"),
            """
            is a machine learning technique where knowledge learned by a model is transferred to a new model.
            This is done in the context of Machine Learning by taking the weights and biases from a trained
            model and using them as initial values for a new model. The new model is then trained on a new
            dataset, which can be much smaller than the original dataset. This technique can be used to quickly
            train new models without having to retrain the entire original model.
            """,
            add_tooltip(references_dict[2]['title'],
                        "3",
                        "ref-3",
                        href=references_dict[2]['href'],
                        ),
        ]),

        html.P(
            "The BERT model is a great example of a model that can be used for transfer learning. BERT was originally trained on a large corpus of English text. We can then use this model to perform tasks such as text classification, question answering, and machine translation."),
        html.P(
            "There are many factors to keep in mind when using transfer learning with the BERT model. In the next sections we will discuss about them in some details."),
        html.Hr()
    ],
    className="text-box card-component row",
)


@callback(

    Output("modal-scroll", "is_open"),
    Input("open-scroll", "n_clicks"),
)
def toggle_text(n1):
    return n1


@callback(
    Output("toggle-bert-offcanvas", "is_open"),
    Input("toggle-bert", "n_clicks"),
)
def toggle_bert(n1):
    return n1


@callback(
    Output("toggle-tl-offcanvas", "is_open"),
    Input("toggle-tl", "n_clicks"),
)
def toggle_tl(n1):
    return n1
