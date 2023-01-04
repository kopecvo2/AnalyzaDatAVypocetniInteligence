import tensorflow as tf
import numpy as np
import pandas as pd

def PCA(X, PC1=1, PC2=2):
    """

    :param X: Matrix m by n with m samples and n features
    :param PC1: Index of first principal component
    :param PC2: Index of second principal component
    :return: H, matrix of principal components
    """
    ConvX = tf.einsum('ji,jk->ik', X, X)
    e, v = tf.linalg.eigh(ConvX)
    sort_order = tf.argsort(e, direction='DESCENDING')
    sort_order = sort_order.numpy()
    v = v.numpy()[:, sort_order]
    H = tf.einsum('ij,jk->ik', X, v)

    names = np.empty(X.shape[0], dtype=object)
    range = np.arange(1, X.shape[0] + 1)
    for number in range:
        names[(number - 1)] = 'PC' + str(number)

    H = pd.DataFrame(tf.transpose(H), columns=names)

    return H, names
