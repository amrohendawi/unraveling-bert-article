import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash import dcc
import plotly.express as px
import plotly.graph_objs as go

from utils import textBox, DATA_PATH, df_to_matrix
from appServer import app
import pandas as pd


def get_data(type):
    df = pd.DataFrame(
        columns=['Model', 'Size', 'Performance', 'Time'])
    if type == "Normal":
        df = df.append(pd.DataFrame([['BERT-base(BERT12)', '100%', '100%', '100%']],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BERT-small(BERT4)', '26%', '91%', '100%']],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        return df
    elif type == "Distillation":
        df = df.append(pd.DataFrame([['DistilBERT(BERT6)', 66, 90, 62]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BERT6-PKD(BERT6)', 62, 98, 52]],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BERT3-PKD(BERT3)', 41, 92, 27]],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['Aguilar(BERT6)', 62, 93, 100]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BERT-48(BERT12)', 1, 87, 1]],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BERT-192(BERTI2)', 17, 93, 4]],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['TinyBERT(BERT4)', 13, 96, 10]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['MobileBERT(BERT24)', 23, 100, 25]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['PD(BERT6)', 62, 98, '40%']], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['WaLDORf(BERT8)', 22, 93, 11]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['MiniLM(BERT6)', 60, 99, 50]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['MiniBERT(mBERT8)', 16, 98, 3]], columns=[
                       'Model', 'Size', 'Performance', 'Time']))
        df = df.append(pd.DataFrame([['BiLSTM-soft(BiLSTMI)', 0.9, 91, 0.2]],
                       columns=['Model', 'Size', 'Performance', 'Time']))
        return df


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
    dcc.Graph(figure=go.Figure(data=[go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")['Size'], name="Size", base=0, width=0.9,),
                                     go.Bar(x=get_data("Distillation")['Model'], y=get_data("Distillation")[
                                            'Performance'], name="Performance", base=0, width=0.6),
                                     go.Bar(x=get_data("Distillation")['Model'], y=get_data(
                                         "Distillation")['Time'], name="Time", base=0, width=0.3),
                                     ],
                               layout=go.Layout(
        barmode='stack',
    )
    )),

    html.Div(html.Img(src=app.get_asset_url('bert-size.png'),
             className="ds_img"), className="img_holder"),
    html.P(" It is often best to train a larger model and then compress it. The benefits of compression are that it can reduce the size of BERT without any impact on downstream tasks. Additionally, compression can make BERT more transferable."),
    html.Hr()
])
