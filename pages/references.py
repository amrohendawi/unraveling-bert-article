from dash import html, dcc
from utils import textBox
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        dcc.Markdown(
            """
            ### References
            """
            , id="references"
        ),
        # a list of bullet points
        dbc.ListGroup(
            [
                dbc.ListGroupItem(
                    "Lundberg, Scott M., and Su-In Lee. \"A unified approach to interpreting model predictions.\" Advances in neural information processing systems 30 (2017).",
                    href="https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
                    style={"display": "list-item"},
                ),
                dbc.ListGroupItem(
                    "Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. \"Attention is all you need.\" Advances in neural information processing systems 30 (2017).",
                    href="https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "S. J. Pan and Q. Yang, \"A Survey on Transfer Learning,\" in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010, doi: 10.1109/TKDE.2009.191.",
                    href="https://ieeexplore.ieee.org/abstract/document/5288526",
                    style={"display": "list-item"},
                ),
                dbc.ListGroupItem(
                    "Vu, Tu, Tong Wang, Tsendsuren Munkhdalai, Alessandro Sordoni, Adam Trischler, Andrew Mattarella-Micke, Subhransu Maji, and Mohit Iyyer. \"Exploring and predicting transferability across NLP tasks.\" arXiv preprint arXiv:2005.00770 (2020).",
                    href="https://arxiv.org/abs/2005.00770",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Kao, Wei-Tsung, Tsung-Han Wu, Po-Han Chi, Chun-Cheng Hsieh, and Hung-Yi Lee. \"BERT's output layer recognizes all hidden layers? Some Intriguing Phenomena and a simple way to boost BERT.\" arXiv preprint arXiv:2001.09309 (2020).",
                    href="https://arxiv.org/abs/2001.09309",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Michel, Paul, Omer Levy, and Graham Neubig. \"Are sixteen heads really better than one?.\" Advances in neural information processing systems 32 (2019).",
                    href="https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Gordon, Mitchell A., Kevin Duh, and Nicholas Andrews. \"Compressing bert: Studying the effects of weight pruning on transfer learning.\" arXiv preprint arXiv:2002.08307 (2020).",
                    href="https://arxiv.org/abs/2002.08307",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Rogers, A., Kovaleva, O. and Rumshisky, A., 2020. A primer in bertology: What we know about how bert works. Transactions of the Association for Computational Linguistics, 8, pp.842-866.",
                    href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Liu, Nelson F., Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. \"Linguistic knowledge and transferability of contextual representations.\" arXiv preprint arXiv:1903.08855 (2019).",
                    href="https://arxiv.org/abs/1903.08855",
                    style={"display": "list-item"},
                ),dbc.ListGroupItem(
                    "Van der Maaten, Laurens, and Geoffrey Hinton. \"Visualizing data using t-SNE.\" Journal of machine learning research 9, no. 11 (2008).",
                    href="https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl",
                    style={"display": "list-item"},
                ),
            ],
            flush=True,
            tag="ol",
            #      make it an ordered list

        ),
    ],
    className="row text-box card-component ",
)
