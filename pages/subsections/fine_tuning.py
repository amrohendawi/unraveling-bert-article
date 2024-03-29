import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output
from utils import read_tasks_nested_tables, df_to_matrix, add_tooltip
from pages.references import references_dict
from appServer import app
import plotly.graph_objs as go
import pandas as pd
import re

fine_tuning_dataframes = read_tasks_nested_tables(
    "2_fine_tuning_taskwise_res_table", df_to_matrix)

# merge all the dictionaries from dict(dict(pd.dataframe)) to dict(pd.dataframe)
fine_tuning_dataframes = {k: pd.concat(v)
                          for k, v in fine_tuning_dataframes.items()}

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
radar_headers = [re.sub(r'([A-Z])', r' \1', header)
                 for header in radar_headers]

# get the tasks from the first dictionary item
tasks = {k: list(d.keys()) for k, d in fine_tuning_dataframes.items()}

text_content = html.Div(
    [
        html.H4("Fine-tuning"),
        html.Br(),
        html.P(
            """
            Fine tuning is the process of tweaking a machine learning model to get better performance. This can involve changes to the
            model architecture, the training data, the training procedure, or any other number of factors. The goal is to improve the
            model's performance on some metric
            """),
        html.P(
            """
            Fine tuning is often an iterative process, where one makes a small change and then evaluates the model's performance on
            a test set. If the performance improves, the change is kept; if not, the change is reverted. This process is repeated until
            the model's performance is satisfactory.
            """),
        html.P(
            """
            Fine tuning is not always necessary, and in some cases, it can actually hurt performance. For example, if a model
            is overfit to the training data, then fine tuning it may just lead to further overfitting. It is important to evaluate
            the model on a hold-out set or cross-validation set before making any changes, to make sure that the changes are actually helping.
            """),
    ],
    id="fine-tuning"
)

content = html.Div([
    text_content,
    html.P(
        """
        Different Fine-tuning techniques can have a different effect on the transferability of BERT models.
        Hyperparameter tuning can help optimize a model for a specific task, while layer freezing can improve the
        stability of a model and prevent overfitting. Feature engineering can also help improve the performance of a
        model by extracting syntactic and semantic features that are relevant to the task at hand.
        """
    ),
    html.P([
        "The following radar chart visualizes the efficacy of applying three different variations of layer freezing"
        "based on Vu et al's work ",
        add_tooltip(references_dict[3]["title"], "4", "ref-4-4", references_dict[3]["href"]),
        ". These are:",
    ]),
    dmc.List(
        [
            dmc.ListItem(
                [
                    html.B("Full-Frozen: "), "Freeze some layers and use full dataset for fine-tuning",
                ]),
            dmc.ListItem(
                [
                    html.B("Full-Unfrozen: "), "Freeze no layer and use full dataset for fine-tuning"
                ]),
            dmc.ListItem(
                [
                    html.B("Limited-Unfrozen: "), "Freeze no layer and use a limited dataset for fine-tuning"
                ]),
        ]),
    html.Br(),
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.P("Pick task domain: ", style={
                            "fontWeight": "bold"}),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="dropdown-task-group",
                            searchable=False,
                            clearable=False,
                            options=[{"label": i, "value": i} for i in
                                     ['1_classification_regression_tasks', '2_Qa_tasks',
                                      '3_sequence_labeling_ft_tasks']],
                            value='1_classification_regression_tasks',
                            className="drop-down-component",
                        ),
                        width=5,
                    ),
                ],
                style={"margin": "auto"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.P("Pick different tasks to compare: ",
                               style={"fontWeight": "bold"}),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="dropdown-task-multiselect",
                            multi=True,
                        ),
                        width=5,
                    ),
                ],
                style={"margin": "auto"},
            ),
        ],
    ),
    html.Br(),
    dcc.Loading(
        dcc.Graph(id="fine_tuning_graph",
                  config={"displayModeBar": False},
                  style={"width": "auto"})
    ), html.Hr()
],
    className="card-component",
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
        title=go.layout.Title(
            text='Tasks transferability comparison based on different fine-tuning methods'),
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
