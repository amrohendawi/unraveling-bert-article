from dash import html, Output, Input, callback
import dash_bootstrap_components as dbc

text_content = html.Div(
    [
        html.H3("BERT and Transfer-Learning"),
        html.Br(),
        html.Div(html.Img(src="assets/transfer-learning.png",
                          className="img_tl",
                          title="https://editor.analyticsvidhya.com/uploads/444861_2vse-G3FKMT2D59NFWduMA.jpg"),
                 className="img_holder"),
        html.P(["Since its release in late 2018, the Bidirectional Encoder Representations from Transformers ",
                html.A(
                    "(BERT OffCanvas)",
                    id="toggle-bert",
                    className="toggle-text",
                ),
                html.A(
                    "(BERT Modal)",
                    id="open-scroll",
                    className="toggle-text",
                ),
                " model has revolutionized the field of Natural Language Processing (NLP). BERT is a transformer-based machine learning model that is trained on a large corpus of text data in a self-supervised manner. BERT has achieved state-of-the-art results on a wide range of NLP tasks, such as text classification, question answering, and text generation. Importantly, BERT can be fine-tuned for specific tasks using just a small amount of training data.",
                ]),
    ],
    id="introduction"
)

layout = html.Div(
    [

        dbc.Tooltip(
            "Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. \"Attention is all you need.\" Advances in neural information processing systems 30 (2017).",
            target="bert-ref",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("BERT")),
                dbc.ModalBody(html.Ul([
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
                        html.A("[2]", id="bert-ref2", href="#references"), ])]), ),
            ],
            id="modal-scroll",
            is_open=False,
        ),
        dbc.Offcanvas(
            html.Ul([
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
                ])]),
            id="toggle-bert-offcanvas",
            title="BERT",
            is_open=False,
            placement="end"
        ),
        dbc.Offcanvas(
            html.Ul([
                html.Li(
                    "Transfer Learning is a technique that can be used to improve the performance of a machine learning model by using knowledge from a related task. The main idea is to transfer knowledge from a model that has been trained on a large and high-quality dataset to a model that is trained on a smaller and lower-quality dataset."),
                html.Li(
                    "The main advantage of transfer learning is that it can be used to train a model on a dataset that is too small or too low quality to train a model from scratch. Additionally, transfer learning can be used to improve the performance of a machine learning model by using knowledge from a related task with less data and computional power."),
                html.Li([
                    "The main disadvantage of transfer learning is that it is difficult to control the amount of knowledge that is transferred from the source model to the target model. Additionally, transfer learning can only be used to improve the performance of a machine learning model if the source and target datasets are similar.",
                    html.A("[3]", id="tl-ref", href="#references"),
                    " One way to avoid this drawback is to freeze some of the weights. We will highlight this technique more in the next sections.",
                ])]),
            id="toggle-tl-offcanvas",
            title="Transfer Learning",
            is_open=False,
            placement="end"
        ),
        text_content,
        html.P([
            html.A(
                "Transfer learning",
                id="toggle-tl",
                className="toggle-text",
            ),
            " is a machine learning technique where knowledge learned by a model is transferred to a new model. This is done in the context of Machine Learning by taking the weights and biases from a trained model and using them as initial values for a new model. The new model is then trained on a new dataset, which can be much smaller than the original dataset. This technique can be used to quickly train new models without having to retrain the entire original model.",
            html.A("[2]", id="bert-ref", href="#references"),
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


# @callback(
#    Output("offcanvas-placement", "is_open"),
#    Input("open-scroll2", "n_clicks"),
# )
# def toggle_text(n1):
#    return n1


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
