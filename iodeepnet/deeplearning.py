import numpy as np


# Activation functions
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def tanh(X):
    return tanh(X)


def softmax(X):
    Y = X - np.max(X)
    return Y / np.sum(np.exp(Y))


# Implementation of Neural Network Architectures:


class BasicRNN():
    def __init__(self, layer_size):
        self.layer_size = layer_size

    def train(self, x_train, y_train, seed=None, epochs=10, callbacks=[]):
        W = {}
        nx, m, Tx = x_train.shape
        a = np.zeros((self.layer_size, m, Tx))
        na = self.layer_size

        # Initialize Parameters Waa, Wax, Wya

        if seed:
            np.random.seed(seed)
        W['aa'] = np.random.randn(na, na)
        W['ax'] = np.random.randn(na, nx)
        W['ya'] = np.random.randn(nx, na)

        for k in W.keys():
            print(W[k][4][1])


class NaiveNeuralNetwork():
    pass
