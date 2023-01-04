
# ## Creating dataset
# import CreateDataset
# import pandas as pd
# import tensorflow as tf
#
# ds = CreateDataset.ReturnDataset()
#
# for row in ds:
#     data = row[0]
#     label = row[1]
#
# X = tf.transpose(data[:, 0, :])
# label = label[:]
# dfX = pd.DataFrame(X)

# Importing from Matlab
import scipy as sp
import tensorflow as tf

label = sp.io.loadmat('C:/Users/vojta/Documents/GitHub/DP_matlab/labels_for_PCA.mat')['labels']
X = tf.transpose(sp.io.loadmat('C:/Users/vojta/Documents/GitHub/DP_matlab/Matrix_for_PCA.mat')['P'])

# PCA
import StatisticalTools as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
# pd.options.plotting.backend = "plotly"
from dash import Dash, dcc, html, Input, Output


H, names = st.PCA(X)
H['label'] = label[0, :]

# Dash
app = Dash(__name__)

app.layout = html.Div([
    html.Div([

        html.Div([dcc.Dropdown(names, 'PC1', id='xaxis')], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([dcc.Dropdown(names, 'PC2', id='yaxis')], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})

    ]),

    dcc.Graph(id='PCA_scatter')

])

@app.callback(
    Output('PCA_scatter', 'figure'),
    Input('xaxis', 'value'),
    Input('yaxis', 'value'))
def update_graph(xaxis_PC, yaxis_PC):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=H[xaxis_PC],
        y=H[yaxis_PC],
        mode='markers',
        marker=dict(color=H.label)
    ))
    fig.update_layout(height=1000)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


# Hello dash

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# from dash import Dash, html, dcc
# import plotly.express as px
# import pandas as pd
#
# app = Dash(__name__)
#
# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
#
# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
#
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for your data.
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)

