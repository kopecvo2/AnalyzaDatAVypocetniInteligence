import numpy as np
import pandas as pd
import StatisticalTools as st
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn_som.som import SOM

def PCA(X):
    """

    :param X: Matrix m by n with m samples and n features
    :return: H, matrix of principal components
    """
    print('starting PCA function')
    ConvX = np.einsum('ji,jk->ik', X, X)
    e, v = np.linalg.eigh(ConvX)
    print('eig computed')
    sort_order = np.argsort(e)
    sort_order = sort_order[::-1]
    v = v[:, sort_order]
    e = e[sort_order]
    print('sorted')
    u = np.dot(np.transpose(v), v)
    # a = np.einsum('ij,jk,kl->il', np.transpose(v), ConvX, v) - np.diag(e)
    a = np.dot(np.transpose(v), ConvX)
    b = np.dot(a, v)
    print('test computed')
    H = np.einsum('ij,jk->ik', X, v)

    names = np.empty(ConvX.shape[0], dtype=object)
    range = np.arange(1, ConvX.shape[0] + 1)
    for number in range:
        names[(number - 1)] = 'PC' + str(number)

    H = pd.DataFrame(H, columns=names)

    return H, names


def PlotPCA(X, label):
    """

    :param X: Matrix m by n with m samples and n features
    :param label: m labels for each feature
    :return: Nothing
    """

    H, names = st.PCA(X)
    H['label'] = label

    # Dash
    app = Dash(__name__)

    app.layout = html.Div(children=[
        html.H1(children='Principal component analysis'),
        html.Div([

            html.Div(children=[
                html.Div(children='Choose PC for x axis'),
                dcc.Dropdown(names, 'PC1', id='xaxis')], style={'width': '48%', 'display': 'inline-block'}),

            html.Div(children=[
                html.Div(children='Choose PC for y axis'),
                dcc.Dropdown(names, 'PC2', id='yaxis')],
                style={'width': '48%', 'float': 'right', 'display': 'inline-block'})

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
        fig.update_xaxes(title=xaxis_PC)
        fig.update_yaxes(title=yaxis_PC)
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
        fig.update_layout(height=800)

        return fig

    # if __name__ == '__main__':
    app.run_server(debug=True)


