import plotly.offline as pyo
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from utils import textBox, read_tasks_nested_tables, DATA_PATH, df_to_matrix
from appServer import app
import plotly.graph_objs as go
import pandas as pd


fine_tuning_dataframes = read_tasks_nested_tables(
    "2_fine_tuning_taskwise_res_table", df_to_matrix)

# merge all the dictionaries from dict(dict(pd.dataframe)) to dict(pd.dataframe)
fine_tuning_dataframes = {k: pd.concat(v) for k, v in fine_tuning_dataframes.items()}

# from the first dictionary item, get the headers of the dataframe
headers = list(list(fine_tuning_dataframes.values())[0].columns)

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

    dcc.Loading(
        dcc.Graph(id="fine_tuning_graph", className="card-component",
                  style={"width": "auto"})
    ),
]
)


content = html.Div([
    fine_tuning_section,
    html.P("Select columns"),
    # dcc.Dropdown(
    #     id="admit-select",
    #     options=[{"label": i, "value": i} for i in admit_list],
    #     value=admit_list[:],
    #     multi=True,
    # ),
    html.Br(),
    html.Div(
        id="reset-btn-outer",
        children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
    ),
], id="fine-tuning"
)

# TODO:
#  add filters:
# 1- ff, fu, lu
# 2- dropdown list t 1,2,3
# 3- dropdown multiselect tasks (11)

@ app.callback(
    Output('fine_tuning_graph', 'figure'),
    Input("dropdown-graph-type", "value"))
def update_figure(list_of_values):

    dummy_data = {
        'data1': [4, 4, 5, 4, 3, 4, 5, 4, 3],
        'data2': [5, 5, 4, 5, 2, 5, 4, 5, 2],
        'data3': [3, 4, 5, 3, 5, 4, 5, 3, 5],
    }

    data = [go.Scatterpolar(r=v, theta=headers, fill='toself', name=k)
            for k, v in dummy_data.items()
    ]
    graph_layout = go.Layout(
        title=go.layout.Title(text='Dummy data comparison'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )

    return go.Figure(data=data, layout=graph_layout)
