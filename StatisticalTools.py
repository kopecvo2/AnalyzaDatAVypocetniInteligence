import numpy as np
import pandas as pd

def PCA(X, PC1=1, PC2=2):
    """

    :param X: Matrix m by n with m samples and n features
    :param PC1: Index of first principal component
    :param PC2: Index of second principal component
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
