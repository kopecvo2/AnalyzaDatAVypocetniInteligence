import numpy as np
import pandas as pd
import StatisticalTools as st
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import somoclu
from sklearn.cluster import KMeans
import sklearn as sk

"""
Vojtech Kopecky, CTU in Prague, 2023
"""

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

    return H


def PlotPCA(X, label, Name='Principal component analysis'):
    """

    :param Name: sting name of the plot
    :param X: Matrix m by n with m samples and n features
    :param label: m labels for each feature
    :return: Nothing
    """

    H = st.PCA(X)
    H['label'] = label

    names = H.columns

    # Dash
    app = Dash(__name__)

    app.layout = html.Div(children=[
        html.H1(children=Name),
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
            marker=dict(color=H.label, line=dict(color='black', width=1))
        ))
        fig.update_xaxes(title=xaxis_PC)
        fig.update_yaxes(title=yaxis_PC)
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
        fig.update_layout(height=800)

        return fig

    # if __name__ == '__main__':
    # app.run_server(debug=False)

    return app


def PlotSOMlarge(X, label):
    """

    :param X: Matrix m by n with m samples and n features
    :param label: m labels for each feature
    :return: Nothing
    """

    n_rows, n_columns = 100, 100
    label0 = np.where(label == 0)[0][0]
    label1 = np.where(label == 1)[0][0]

    som = somoclu.Somoclu(n_columns, n_rows, data=X)
    print('SOM training started')
    som.train(epochs=50)

    print('plotting U matrix with BMU')
    som.view_umatrix(bestmatches=True, labels=label)

    print('plotting activation map of first sample with label 0')
    som.view_activation_map(None, label0, bestmatches=True, labels=label)

    print('plotting activation map of first sample with label 1')
    som.view_activation_map(None, label1, bestmatches=True, labels=label)


def PlotSOM4(X, label):
    """

    :param X: Matrix m by n with m samples and n features
    :param label: m labels for each feature
    :return: Nothing
    """

    n_rows, n_columns = 2, 2
    label0 = np.where(label == 0)[0][0]
    label1 = np.where(label == 1)[0][0]

    som = somoclu.Somoclu(n_columns, n_rows, data=X)
    print('SOM training started')
    som.train(epochs=50)

    print('plotting U matrix with BMU')
    som.view_umatrix(bestmatches=True, labels=label)

    print('plotting activation map of first sample with label 0')
    som.view_activation_map(None, label0, bestmatches=True, labels=label)

    print('plotting activation map of first sample with label 1')
    som.view_activation_map(None, label1, bestmatches=True, labels=label)

    init_df = np.zeros((2, 4))
    bin_df = pd.DataFrame(init_df, index=['num_of_zero_labeled', 'num_of_one_labeled'],
                          columns=['bin_00', 'bin_01', 'bin_10', 'bin_11'])
    i = 0
    for bmu in som.bmus:
        if np.all(bmu == np.array([0, 0])):
            usebin = 'bin_00'
        elif np.all(bmu == np.array([0, 1])):
            usebin = 'bin_01'
        elif np.all(bmu == np.array([1, 0])):
            usebin = 'bin_10'
        elif np.all(bmu == np.array([1, 1])):
            usebin = 'bin_11'
        else:
            print('BMU sorting Error!!!')
            usebin = 'bin_error'

        if label[i] == 0:
            bin_df[usebin][0] += 1
        elif label[i] == 1:
            bin_df[usebin][1] += 1
        else:
            print('BMU sorting Error!!!')

        i += 1

    print('Sample separation into BMU bins with labels')
    print('------------------------------------------------------')
    print(bin_df)
    print('------------------------------------------------------')


def PlotKmeans(X, label):

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(X)

    init_df = np.zeros((2, 2))
    bin_df = pd.DataFrame(init_df, index=['num_of_zero_labeled', 'num_of_one_labeled'],
                          columns=['bin_0', 'bin_1'])

    # Plot in PCA
    app = PlotPCA(X, kmeans.labels_, Name='Kmeans clusters in PCA space')

    i = 0
    for prediction in kmeans.labels_:
        if np.all(prediction == np.array(0)):
            usebin = 'bin_0'
        elif np.all(prediction == np.array(1)):
            usebin = 'bin_1'
        else:
            print('BMU sorting Error!!!')
            usebin = 'bin_error'

        if label[i] == 0:
            bin_df[usebin][0] += 1
        elif label[i] == 1:
            bin_df[usebin][1] += 1
        else:
            print('BMU sorting Error!!!')

        i += 1

    print('Sample separation into BMU bins with labels')
    print('------------------------------------------------------')
    print(bin_df)
    print('------------------------------------------------------')

    return app


def PlotTSNE(X, label):
    tsne = sk.manifold.TSNE(n_components=2, perplexity=4)  # perplexity was 50
    print('fitting tsne')
    tsne.fit(X)
    print('tsne computed')

    H = tsne.embedding_

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=H[:, 0],
        y=H[:, 1],
        mode='markers',
        marker=dict(color=label)
    ))
    fig.show()
