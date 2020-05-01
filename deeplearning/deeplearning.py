import numpy as np
import helper_funcs as hf


# Activation functions


# Implementation of Neural Network Architectures:


class BasicRNN:
    def __init__(self, layer_size):
        self.na = layer_size
        self.parameters = {}
        self.cells = []  # List of the cells in the network

    # Cell class for basic RNN network. Class will be generalized to include GRU and LSTM, will be a standalone Class.
    class Cell:
        def __init__(self, na, nx, ny):
            '''
            # Create placeholder for the Cell parameters, inputs, and outputs
            # Cell parameters: Waa, Wax, Wya, ba, by  (stored in dictionary)
            # Cell input  a_prev, x
            # Cell output a_next, y_predict
            '''

            self.params = {}  # Placeholder dictionary for the cell parameters
            self.na = na
            self.nx = nx
            self.ny = ny
            self.a_prev = np.array([])
            self.a_next = np.array([])
            self.x = np.array([])
            self.y_predict = np.array([])
            self.__initialize_parameters(self)  # Initializes cell parameters with random numbers.

        def __initialize_parameters(self, seed=None):
            self.params['Waa'] = np.random.randn(self.na, self.na)
            self.params['Wax'] = np.random.randn(self.na, self.nx)
            self.params['Wya'] = np.random.randn(self.ny, self.na)
            self.params['ba'] = np.random.randn(self.na, 1)
            self.params['by'] = np.random.randn(self.ny, 1)

        def get_params(self):
            return self.params

        def forward(self, a_prev, X):
            self.a_prev = a_prev
            self.a_next = hf.tanh(np.dot(self.params['Waa'], a_prev) +
                                  np.dot(self.params['Wax'], X) +
                                  self.params['ba'])
            self.y_predict = hf.softmax(np.dot(self.params['Wya'], self.a_next) + self.params['by'])

            return self.a_next, self.y_predict

        def backward(self):
            return

    def train(self, X_train, Y_train, epochs=10):
        nx, m, Tx = X_train.shape
        ny, m, Ty = Y_train.shape
        y_predict = np.zeros((ny, m, Ty))

        # Create the network, with Tx RNN cells.
        for i in range(Tx):
            self.cells.append(self.Cell(self.na, nx, ny))

        # Initial activation vector zeros.
        a0 = np.random.randn(self.na, m)
        a = np.zeros((self.na, m, Tx))

        a[:, :, 0], y_predict[:, :, 0] = self.cells[0].forward(a0, X_train[:, :, 0])

        # Start training loop for epoch number of times, or until any other given criteria is met.
        for t in range(Tx - 1):
            a[:, :, t + 1], y_predict[:, :, t + 1] = self.cells[t + 1].update(a[:, :, t], X_train[:, :, t])
