
## Creating dataset from simulated results
# import CreateDataset
# import pandas as pd
# import tensorflow as tf
# import StatisticalTools as st
# import sklearn as sk
#
# ds = CreateDataset.ReturnDataset()
#
# for row in ds:
#     data = row[0]
#     label = row[1]
#
# X = data[:, 0, :]
# label = label[:]
# dfX = pd.DataFrame(X)
#
# # st.PlotSOM4(X, label)
# #
# # app = st.PlotKmeans(X, label)
# # app.run_server()
#
# st.PlotTSNE(X, label)

## Importing from Matlab
import scipy as sp
import StatisticalTools as st
import sklearn as sk
from sklearn.cluster import KMeans
import plotly.graph_objects as go

label = sp.io.loadmat('C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/AnalyzaDatAVypocetniInteligence/From_DP_matlab/labels_for_PCA.mat')['labels'][0, :]
X = sp.io.loadmat('C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/AnalyzaDatAVypocetniInteligence/From_DP_matlab/Matrix_for_PCA.mat')['P']

app = st.PlotPCA(X, label)

st.PlotSOMlarge(X, label)

st.PlotSOM4(X, label)

st.PlotTSNE(X, label)

app.run_server()

## Importing aluminum die casting data - all eigenshapes in one matrix

# import numpy as np
# import scipy as sp
# import StatisticalTools as st
#
# X = sp.io.loadmat('C:/Users/vojta/Documents/GitHub/AnalyzaDatAVypocetniInteligence/input_data_aluminum_casting/PCA_XX_z.mat')['XX_z']
# label = np.array([1, 1, 0, 0, 0])   # Improvise - labels are not known
#
# app = st.PlotPCA(X, label)
# app.run_server()

# st.PlotTSNE(X, label)
#
# st.PlotSOM4(X, label)
#
# app = st.PlotKmeans(X, label)
# app.run_server()


## Importing aluminum die casting data - one eigenvector in matrix

# import numpy as np
# import scipy as sp
# import StatisticalTools as st
#
# X = sp.io.loadmat('C:/Users/pc/OneDrive - České vysoké učení technické v Praze/DATA_D/_GithubProjectData/AnalyzaDatAVypocetniInteligence/input_data_aluminum_casting/PCA_X_z1.mat')['X_z']
# # label = np.array([0, 0, 1, 0, 1])   # Improvise - labels are not known (labels for first eigenvector)
# label = np.array([1, 0, 0, 0, 1])   # Improvise - labels are not known (labels for eigenvector 2, 3, 4, 5, 6, 7)
#
# app = st.PlotPCA(X, label)
# app.run_server()              # This version looks good


## Importing aluminum die casting data - MAC criterion diagonals in matrix X

# import numpy as np
# import scipy as sp
# import StatisticalTools as st
#
# X = sp.io.loadmat('C:/Users/vojta/Documents/GitHub/AnalyzaDatAVypocetniInteligence/input_data_aluminum_casting/MAC_PCA_X.mat')['diagonals']
# label = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1])   # Improvise - labels are not known
#
# app = st.PlotPCA(X, label)
# app.run_server()

# st.PlotTSNE(X, label)

## Plotting

# st.PlotPCA(X, label)


## SOM
# from sklearn_som.som import SOM
#
# import somoclu
#
# n_rows, n_columns = 20, 20
#
# # c1 = np.random.rand(50, 3)/5
# # c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
# # c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
# # data = np.float32(np.concatenate((c1, c2, c3)))
# data = X
# data_vec = data[0:1, :]
# labels = label
# # colors = ["red"] * 50
# # colors.extend(["green"] * 50)
# # colors.extend(["blue"] * 50)
# # labels = range(150)
#
#
# i = 0
# colors = []
# for lab in labels:
#     if lab>0:
#         colors.extend(["red"])
#     else:
#         colors.extend(["green"])
#     i += 1
#
# ##
#
# som = somoclu.Somoclu(n_columns, n_rows, data=data)
# ##
#
# som.train(epochs=50)
# ##
# #som.view_component_planes()
#
#
# ##
# som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)
#
# som.view_activation_map(data_vector=data_vec, labels=labels)
#
# # som.view_activation_map(None, 0, labels=labels)
# # som.view_activation_map(None, 1, labels=labels)
# # som.view_activation_map(None, 2, labels=labels)
# # som.view_activation_map(None, 3, labels=labels)
# # som.view_activation_map(None, 4, labels=labels)
# # som.view_activation_map(None, 5, labels=labels)
# # som.view_activation_map(None, 6, labels=labels)
#
#
# # som.cluster()
#
# som.view_umatrix(bestmatches=True)



# ## PCA
# import StatisticalTools as st
# import numpy as np
# import plotly.graph_objects as go
# import pandas as pd
# # pd.options.plotting.backend = "plotly"
# from dash import Dash, dcc, html, Input, Output
#
#
# H, names = st.PCA(X)
# H['label'] = label[:]
#
# # Dash
# app = Dash(__name__)
#
# app.layout = html.Div(children=[
#     html.H1(children='Principal component analysis'),
#     html.Div([
#
#         html.Div(children=[
#             html.Div(children='Choose PC for x axis'),
#             dcc.Dropdown(names, 'PC1', id='xaxis')], style={'width': '48%', 'display': 'inline-block'}),
#
#         html.Div(children=[
#             html.Div(children='Choose PC for y axis'),
#             dcc.Dropdown(names, 'PC2', id='yaxis')], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
#
#     ]),
#
#     dcc.Graph(id='PCA_scatter')
#
# ])
#
# @app.callback(
#     Output('PCA_scatter', 'figure'),
#     Input('xaxis', 'value'),
#     Input('yaxis', 'value'))
# def update_graph(xaxis_PC, yaxis_PC):
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=H[xaxis_PC],
#         y=H[yaxis_PC],
#         mode='markers',
#         marker=dict(color=H.label)
#     ))
#     fig.update_xaxes(title=xaxis_PC)
#     fig.update_yaxes(title=yaxis_PC)
#     fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
#     fig.update_layout(height=800)
#
#     return fig
#
# if __name__ == '__main__':
#     app.run_server(debug=True)


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
##
# import numpy as np
# A = np.array([[1, 2, 3],[2, 1, 4],[3, 4, 5]])
# e, v = np.linalg.eig(A)
#
# u = np.dot(np.transpose(v), A)
# l = np.dot(u, v)

