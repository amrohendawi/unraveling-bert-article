import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from utils import textBox, read_tasks_nested_tables, df_to_matrix
from appServer import app
import plotly.graph_objs as go
import pandas as pd
import re

fine_tuning_dataframes = read_tasks_nested_tables(
    "2_fine_tuning_taskwise_res_table", df_to_matrix)

# merge all the dictionaries from dict(dict(pd.dataframe)) to dict(pd.dataframe)
fine_tuning_dataframes = {k: pd.concat(v) for k, v in fine_tuning_dataframes.items()}

# for every task group and epoch, draw a radar graph
for task_group, df in fine_tuning_dataframes.items():
    # transpose the dataframe
    df = df.T
    cols = df.columns
    # update df column names by the second part of the cols tuples
    df.columns = [col[1] for col in cols]
    fine_tuning_dataframes[task_group] = df

# get the first dictionary item
first_dict = list(fine_tuning_dataframes.values())[0]

# get the row headers of the dataframe
radar_headers = first_dict.index.tolist()
# add space before every uppercase letter in the row headers
radar_headers = [re.sub(r'([A-Z])', r' \1', header) for header in radar_headers]

# get the tasks from the first dictionary item
tasks = {k: list(d.keys()) for k, d in fine_tuning_dataframes.items()}

fine_tuning_section = html.Div([
    textBox(
        """
            #### Fine-tuning
            When a model is fine-tuned on a specific task, its transferability to other tasks is usually enhanced.
            This is because the model has been specifically optimized for the task at hand, and so is better able to generalize to other tasks.

            There is some evidence that fine-tuning can also improve a model's ability to transfer to other domains.
            For example, a model that is fine-tuned on a medical domain may be able to better transfer to other medical domains.
            However, it is not clear how much of an improvement fine-tuning provides in this case.
            """, text_id="fine-tuning"
    ),
    dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id="dropdown-task-group",
                    searchable=False,
                    clearable=False,
                    options=[{"label": i, "value": i} for i in
                             ['1_classification_regression_tasks', '2_Qa_tasks', '3_sequence_labeling_ft_tasks']],
                    value='1_classification_regression_tasks',
                    className="drop-down-component",
                )
            ),
            dcc.Dropdown(
                id="dropdown-task-multiselect",
                multi=True,
            ),
        ]
    ),
    html.Br(),
    dcc.Loading(
        dcc.Graph(id="fine_tuning_graph", className="card-component",
                  style={"width": "auto"})
    ),
]
)

content = html.Div([
    fine_tuning_section,
]
)


@app.callback(
    [
        Output('dropdown-task-multiselect', 'options'),
        Output('dropdown-task-multiselect', 'value'),
    ],
    Input('dropdown-task-group', 'value')
)
def update_tasks_dropdown_multiselect(task_group):
    return [{'label': i, 'value': i} for i in tasks[task_group]], tasks[task_group][:3]


@app.callback(
    Output('fine_tuning_graph', 'figure'),
    [
        Input("dropdown-task-group", "value"),
        Input("dropdown-task-multiselect", "value"),
    ])
def update_figure(task_group, tasks_multiselect):
    selected_tasks = fine_tuning_dataframes[task_group][tasks_multiselect]
    selected_tasks = selected_tasks.to_dict('list')

    data = [go.Scatterpolar(r=v, theta=radar_headers, fill='toself', name=k)
            for k, v in selected_tasks.items()
            ]
    graph_layout = go.Layout(
        title=go.layout.Title(text='Tasks transferability comparison based on different fine-tuning methods'),
        showlegend=True,
    )
    fig = go.Figure(data=data, layout=graph_layout)
    fig.update_layout(
        legend_title="Task",
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'autosize': True,
    })
    return fig
