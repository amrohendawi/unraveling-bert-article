from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div(
    [
        html.H3("Author Contributions", id="author-contributions"),
        html.Br(),
        dcc.Markdown(
            """
            **Amro Hendawi** came up with the main idea, spearheaded the story, the sections and the structure of the project.
            **Ammer Ayach** helped in the research and aggregation of academic sources. He also collected/generated most
            of the data used in the project.
            **Amro Hendawi** also designed the website and made it available to the public. He also used the data to create
            the visualizations.
            **Thomas Goerttler** supervised the project and oversaw the development of the project. He also provided feedback
            and guidance to the team. 
            """
        ),
    ],
    className="row text-box card-component",
)
