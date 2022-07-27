import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objs as go

from utils import DATA_PATH
from appServer import app
import pandas as pd


def append_frames(list):
    df = pd.DataFrame(columns=['Model', 'Size', 'Performance', 'Time'])
    df = pd.concat(
        [df, pd.DataFrame(list, columns=['Model', 'Size', 'Performance', 'Time'])])
    return df


scatter_plot_data = pd.read_csv(DATA_PATH.joinpath("bert_compression_results_modified.csv"))

# scatter_plot_data = pd.read_csv(DATA_PATH.joinpath("bert_compression_results.csv"))
# # convert Performance column from string of format "99%" to float between 0 and 1
# scatter_plot_data['Performance'] = scatter_plot_data['Performance'].apply(
#     lambda x: int(x[:-1]) / 100)
# # create a new column called #Parameters that is the number cut from the end of the model name
# scatter_plot_data['#Parameters'] = scatter_plot_data['Model'].apply(
#     lambda x: int(x.split('BERT')[-1]) if 'BERT' in x else 93)
# model_parameters_dict = {
#     3: 45.7,
#     4: 53.2,
#     6: 67,
#     12: 110,
#     24: 340,
# }
# # convert the #Parameters column using the model_parameters_dict dictionary
# scatter_plot_data['#Parameters'] = scatter_plot_data['#Parameters'].apply(
#     lambda x: model_parameters_dict[x] if x in model_parameters_dict.keys() else x)
# scatter_plot_data['ParametersAfterCompression'] = round(
#     scatter_plot_data['#Parameters'] / scatter_plot_data['Compression'], 4)
# scatter_plot_data.to_csv(DATA_PATH.joinpath("bert_compression_results_modified.csv"), index=False)


def get_data(type):
    if type == "Normal":
        data = [['BERT-base (BERT12)', 100, 100, 100],
                ['BERT-small (BERT4)', 26, 91, 100]]
        return append_frames(data)
    elif type == "Distillation":
        data = [['DistilBERT (BERT6)', 66, 90, 62], ['BERT6-PKD (BERT6)', 62, 98, 52],
                ['BERT3-PKD (BERT3)', 41, 92, 27],
                ['Aguilar (BERT6)', 62, 93, 100], ['BERT-48 (BERT12)', 1, 87, 1], [
                    'BERT-192 (BERTI2)', 17, 93, 4], ['TinyBERT (BERT4)', 13, 96, 10],
                ['MobileBERT (BERT24)', 23, 100, 25], ['PD (BERT6)', 62, 98, '40%'], ['WaLDORf (BERT8)', 22, 93, 11],
                ['MiniLM (BERT6)', 60, 99, 50], ['MiniBERT (mBERT8)', 16, 98, 3],
                ['BiLSTM-soft (BiLSTMI)', 0.9, 91, 0.2]]
        return append_frames(data)
    elif type == "Quantization":
        data = [['Q-BERT-MP (BERT12)', 7, 98, 100], ['BERT-QAT (BERT12)',
                                                     25, 99, 100], ['GOBO (BERT12)', 10, 99, 100]]
        return append_frames(data)
    elif type == "Pruning":
        data = [['McCarley (BERT24)', 45, 98, 52], ['RPP (BERT24)', 58, 99, 100], [
            'Soft MVP (BERT12)', 3, 94, 100], ['IMP (BERT12)', 40, 94, 100]]
        return append_frames(data)
    elif type == "Other":
        data = [['ALBERT-base (BERT12)', 11, 97, 100], ['ALBERT-xxlarge (BERT12)', 200, 107, 100], [
            'BERT-of-Theseus (BERT6)', 62, 98, 52], ['PoWER-BERT (BERT12)', 100, 99, 22]]
        return append_frames(data)


scatter_plot_fig = px.scatter(
    scatter_plot_data,
    x="Compression",
    y="Performance",
    color="Method",
    size="#Parameters",
    trendline="ols",
    hover_data={
        "Name": True,
        "Performance": True,
        "Speedup": True,
        "#Parameters": True,
        "Compression": True,
        "Method": False,
    },

)

scatter_plot_fig.update_layout({
    "title": "BERT Compression Performance",
    "xaxis": {
        "title": "Compression Factor",
        "type": "log",

    },
    "yaxis": {
        "title": "Performance",
        "type": "log",
    },
    'plot_bgcolor': 'rgba(77, 137, 110, 0.22)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'autosize': True,
})

scatter_plot_fig.update_yaxes(tickformat='.0%')
# convert x axis to a factor scale such as 1x, 2x, 3x, 4x, etc.
# scatter_plot_fig.update_xaxes(
#     #     for any number convert it to a string of format "1x", "2x", "3x", "4x", etc.
#     tickvals=scatter_plot_data['Compression'],
#     ticktext=scatter_plot_data['Compression'].apply(lambda x: str(x) + "x"),
# #     keep it logarithmic
#     type="log",
#
# )

# update traces marker line width by the compression factor for each trace
scatter_plot_fig.update_traces(marker=dict(
    line=dict(width=scatter_plot_data['Compression'])
))

scatter_plot = html.Div(
    [
        dcc.Graph(
            id='model-size-graph',
            figure=scatter_plot_fig,
            config={"displayModeBar": False},
        ),
    ],
    className="card-component", style={"width": "fit-content"}
)

text_content = html.Div(
    [
        html.H4("How big should BERT be?"),
        html.P(
            "The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
        html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
                html.A("[5]", id="t5-ref", href="#references")]),
        html.P([
            "The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
            html.A("[6", id="t6-ref", href="#references")]),  # TODO: elaborate a little on the texts here
        html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
                html.A("[7]", id="t7-ref", href="#references")]),
    ],
    id="model-size"
)

content = html.Div([
    text_content,
    html.P([
        "The following table shows many version of Tranformer with many options of comparsion and the related speed up. The values of size, performance and time are against the BERT base in percent",
        html.A("[8]", id="t8-ref", href="#references")]),
    # TODO: make toggle and hard-code the rest of table

    html.Div([
        dcc.Dropdown(['Normal', 'Distillation', 'Quantization', 'Pruning',
                      'Other'], 'Normal', id='sizeplot-dropdown', clearable=False),
        # TODO: add callback to display content of each technique and add advance reader to each technique
        html.P("", id="depth_text")

    ]),
    dcc.Graph(figure=go.Figure(
        data=[go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Size'], name="Size", base=0, width=0.9, ),
              go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")[
                  'Performance'], name="Performance", base=0, width=0.6),
              go.Bar(x=get_data("Normal")['Model'], y=get_data(
                  "Normal")['Time'], name="Time", base=0, width=0.3),
              ],
        layout=go.Layout(
            # yaxis = list(tickformat = "%"),
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={"ticksuffix": "%"}
        ),
    ).update_yaxes(gridcolor='Black'),
              id="size_plot", ),
    html.P(
        " It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    scatter_plot,
    html.Hr()
])


@app.callback(
    Output('depth_text', 'children'),
    Input('sizeplot-dropdown', 'value')
)
def update_textt_size(value):
    if value == "Quantization":
        return "Quantization is a technique used to make the model smaller"
    elif value == "Distillation":
        return "Distillation is a technique used to make the model smaller"
    elif value == "Other":
        return ""
    elif value == "Pruning":
        return "Pruning is a technique used to make the model smaller"
    else:
        return "BERT is relassed in base and small style"


@app.callback(
    Output('size_plot', 'figure'),
    Input('sizeplot-dropdown', 'value')
)
def update_output_size(value):
    if value == "Quantization":
        return go.Figure(data=[
            go.Bar(x=get_data("Quantization")['Model'], y=get_data("Quantization")['Size'], name="Size", base=0,
                   width=0.9, ),
            go.Bar(x=get_data("Quantization")['Model'], y=get_data("Quantization")[
                'Performance'], name="Performance", base=0, width=0.6),
            go.Bar(x=get_data("Quantization")['Model'], y=get_data(
                "Quantization")['Time'], name="Time", base=0, width=0.3),
        ], layout=go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', yaxis={"ticksuffix": "%"})).update_yaxes(
            gridcolor='Black')
    elif value == "Distillation":
        return go.Figure(data=[
            go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")['Size'], name="Size", base=0,
                   width=0.9, ),
            go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")[
                'Performance'], name="Performance", base=0, width=0.6),
            go.Bar(x=get_data("Distillation")['Model'], y=get_data(
                "Distillation")['Time'], name="Time", base=0, width=0.3),
        ], layout=go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', yaxis={"ticksuffix": "%"})).update_yaxes(
            gridcolor='Black')
    elif value == "Other":
        return go.Figure(
            data=[go.Bar(x=get_data("Other")['Model'], y=get_data("Other")['Size'], name="Size", base=0, width=0.9, ),
                  go.Bar(x=get_data("Other")['Model'], y=get_data("Other")[
                      'Performance'], name="Performance", base=0, width=0.6),
                  go.Bar(x=get_data("Other")['Model'], y=get_data(
                      "Other")['Time'], name="Time", base=0, width=0.3),
                  ],
            layout=go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', yaxis={"ticksuffix": "%"})).update_yaxes(
            gridcolor='Black')
    elif value == "Pruning":
        return go.Figure(data=[
            go.Bar(x=get_data("Pruning")['Model'], y=get_data("Pruning")['Size'], name="Size", base=0, width=0.9, ),
            go.Bar(x=get_data("Pruning")['Model'], y=get_data("Pruning")[
                'Performance'], name="Performance", base=0, width=0.6),
            go.Bar(x=get_data("Pruning")['Model'], y=get_data(
                "Pruning")['Time'], name="Time", base=0, width=0.3),
        ], layout=go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', yaxis={"ticksuffix": "%"})).update_yaxes(
            gridcolor='Black')
    else:
        return go.Figure(
            data=[go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Size'], name="Size", base=0, width=0.9, ),
                  go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")[
                      'Performance'], name="Performance", base=0, width=0.6),
                  go.Bar(x=get_data("Normal")['Model'], y=get_data(
                      "Normal")['Time'], name="Time", base=0, width=0.3),
                  ],
            layout=go.Layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', yaxis={"ticksuffix": "%"})).update_yaxes(
            gridcolor='Black')
