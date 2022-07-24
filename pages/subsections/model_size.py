from re import I
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go

from utils import textBox, DATA_PATH, df_to_matrix
from appServer import app
import pandas as pd

def append_frames(list):
       df = pd.DataFrame(columns=['Model', 'Size', 'Performance', 'Time'])
       df = pd.concat([df, pd.DataFrame(list,columns=['Model', 'Size', 'Performance', 'Time'])])
       return df

def get_data(type):
    
    if type == "Normal":
       data = [['BERT-base (BERT12)', 100, 100, 100],['BERT-small (BERT4)', 26, 91, 100]]
       return append_frames(data) 
    elif type == "Distillation":
        data = [['DistilBERT (BERT6)', 66, 90, 62], ['BERT6-PKD (BERT6)', 62, 98, 52],['BERT3-PKD (BERT3)', 41, 92, 27], 
        ['Aguilar (BERT6)', 62, 93, 100], ['BERT-48 (BERT12)', 1, 87, 1], ['BERT-192 (BERTI2)', 17, 93, 4],['TinyBERT (BERT4)', 13, 96, 10],
        ['MobileBERT (BERT24)', 23, 100, 25], ['PD (BERT6)', 62, 98, '40%'], ['WaLDORf (BERT8)', 22, 93, 11], ['MiniLM (BERT6)', 60, 99, 50], ['MiniBERT (mBERT8)', 16, 98, 3], ['BiLSTM-soft (BiLSTMI)', 0.9, 91, 0.2]]
        return append_frames(data) 
    elif type == "Quantization":
        data = [['Q-BERT-MP (BERT12)', 7, 98, 100], ['BERT-QAT (BERT12)', 25, 99, 100], ['GOBO (BERT12)', 10, 99, 100]]
        return append_frames(data) 
    elif type == "Pruning":
        data = [['McCarley (BERT24)', 45 ,98,52],['RPP (BERT24)', 58,99,100],['Soft MVP (BERT12)', 3,94,100],['IMP (BERT12)', 40,94,100]]
        return append_frames(data) 
    elif type == "Other":
        data = [['ALBERT-base (BERT12)', 11,97,100],['ALBERT-xxlarge (BERT12)', 200,107,100],['BERT-of-Theseus (BERT6)', 62,98,52],['PoWER-BERT (BERT12)', 100,99,22]]
        return append_frames(data) 

content = html.Div([
    html.H3("How big should BERT be?"),

    html.P("The size of the BERT model has a significant impact on the performance and the time required to complete the task."),
    html.P(["Too many BERT heads and layers can be harmful to the performance of downstream tasks. ",
           html.A("[5]", id="t5-ref", href="#references")]),
    html.P(["The disabling of certain heads in the architecture had a positive effect on machine translation and abstractive summarization. ",
           html.A("[6", id="t6-ref", href="#references")]),  # TODO: elaborate a little on the texts here
    html.P(["30-40 percent of weights can be pruned without any impact on downstream tasks. ",
           html.A("[7]", id="t7-ref", href="#references")]),
    html.P(["The following tabel shows many version of Tranformer with many options of comparsion and the related speed up. The values of size, performance and time are against the BERT base in percent",
           html.A("[8]", id="t8-ref", href="#references")]),
       #TODO: make toggle and hard-code the rest of tabel 

        html.Div([
    dcc.Dropdown(['Normal','Distillation', 'Quantization', 'Pruning', 'Other'], 'Normal', id='sizeplot-dropdown',clearable=False),
    html.P("", id="depth_text") #TODO: add callback to display content of each technique and add advance reader to each technique
   
]),
    dcc.Graph(figure=go.Figure(data=[go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Normal")['Model'], y=get_data( "Normal")['Time'], name="Time", base=0, width=0.3),
                                     ],
                               layout=go.Layout(
                                   #yaxis = list(tickformat = "%"),
                                   barmode='stack',
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis={"ticksuffix" : "%"}
    ),
    ).update_yaxes(gridcolor='Black'),
    id = "size_plot",),
    html.P(" It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
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
#
#@app.callback(
#    Output('size_plot', 'figure'),
#    Input('sizeplot-dropdown', 'value')
#)
def update_output_size(value):
       print(value)
       if value == "Quantization":
              return go.Figure(data=[go.Bar(x=get_data("Quantization")['Model'], y=get_data("Quantization")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Quantization")['Model'], y=get_data("Quantization")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Quantization")['Model'], y=get_data( "Quantization")['Time'], name="Time", base=0, width=0.3),
                                     ],layout=go.Layout(barmode='stack',plot_bgcolor='rgba(0,0,0,0)')).update_yaxes(gridcolor='Black')
       elif value == "Distillation":
              return go.Figure(data=[go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")['Time'], name="Time", base=0, width=0.3),
                                     ],layout=go.Layout(barmode='stack',plot_bgcolor='rgba(0,0,0,0)')).update_yaxes(gridcolor='Black')
       elif value == "Other":
              return go.Figure(data=[go.Bar(x=get_data("Other")['Model'], y=get_data("Other")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Other")['Model'], y=get_data("Other")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Other")['Model'], y=get_data("Other")['Time'], name="Time", base=0, width=0.3),
                                     ],layout=go.Layout(barmode='stack',plot_bgcolor='rgba(0,0,0,0)')).update_yaxes(gridcolor='Black')
       elif value == "Pruning":
              return go.Figure(data=[go.Bar(x=get_data("Pruning")['Model'], y=get_data("Pruning")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Pruning")['Model'], y=get_data("Pruning")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Pruning")['Model'], y=get_data("Pruning")['Time'], name="Time", base=0, width=0.3),
                                     ],layout=go.Layout(barmode='stack',plot_bgcolor='rgba(0,0,0,0)')).update_yaxes(gridcolor='Black')
       else:
              return go.Figure(data=[go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Normal")['Model'], y=get_data("Normal")['Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Normal")['Model'], y=get_data( "Normal")['Time'], name="Time", base=0, width=0.3),
                                     ],layout=go.Layout(barmode='stack',plot_bgcolor='rgba(0,0,0,0)',),
    ).update_yaxes(gridcolor='Black')
    #
    #elif value == "Pruning":
    #
    # else value == "Distillation"
    