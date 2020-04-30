import numpy as np


# Activation functions
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


# Implementation of Neural Network Architectures:


class BasicRNN:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.parameters  = {}

#   Cell class for basic RNN network. Class will be generalized to include GRU and LSTM, will be a standalone Class.
    class Cell:
        def __init__(self, na, nx, ny):
            self.params = {}
            self.na = na
            self.nx = nx
            self.ny = ny

        def __initialize_parameters(self):
            self.na = a_prev.shape[0]
            self.params['Waa']  = np.random.randn(self.na, self.na)
            self.params['Wax']  = np.random.randn(self.na, self.nx)
            self.params['Wya']  = np.random.randn(self.ny, self.nx)
            self.params['ba']   = np.random.randn(self.na, 1)
            self.params['by']   = np.random.randn(self.ny, 1)

        def get_parameters():
            return self.params


        def update(self, a_prev):


        def run_cell(self):
            return

    def train(self, X, Y, epochs=1000):
        nx, m, Tx = X.shape
        ny, m, Ty = Y.shape
        a0 = np.random.randn(self.na,m)


        # Create the network, with Tx RNN cells.
        for i in range(Tx):
            cells.append(Cell(self.layer_size, nx, ny))


        cells = []
        a[:, :, 0] = cells[0].update(a0, X)

        a[:, :, 0] = np.random.randn(self.na, m)

        for epoch in epochs:
            for t in range(Tx):
                a[:, :, t+1] = cells[t].update(a[:, :, t], X[:, :, t])


    def __forward_propagate(self):
        # Forward propagation through T<X>

        for t in range(Tx-1):
            self.a[:, :, t+1] = tanh(np.dot(self.parameters['Waa'], self.a[:, :, t]) +
                                     np.dot(self.parameters['Wax'], x_train[:, :, t+1]) +
                                     np.dot(self.parameters['ba'])

            y_predict[:, :, t+1] = softmax(np.dot(self.parameters['Wya'], a[:, :, t+1]) +
                                           self.parameters['by'])


            self.caches.append(cache)
            return y_predict

    def train(self, x_train, y_train, seed=None, epochs=10, callbacks=[]):

        # Get the required dimensions for training
        nx, m, Tx = x_train.shape
        ny, Ty = y_train.shape[0], y_train.shape[0]

        self.__initialize_parameters(na, nx, ny, m, Tx)

        y_predict = np.zeros((ny, m, Ty))

        na = self.layer_size



class NaiveNeuralNetwork():
    pass
