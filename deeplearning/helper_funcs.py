import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def tanh(X):
    """

    :rtype: object
    """
    return np.tanh(X)


def softmax(X):
    Y = X - np.max(X)
    return Y / np.sum(np.exp(Y))