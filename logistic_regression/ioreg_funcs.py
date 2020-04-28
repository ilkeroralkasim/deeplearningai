import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_prop(W, A_prev, b):

    """
    :param W:  Parameter matrix, W
    :param A:  Input, activation of the previous layers
    :param b:  Bias vector
    :return:   Z = W * A + b
    """
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
    return A

def initialize_params(dims,  initializer='zeros', seed=None):
    """
    :param dims: dimensions of the W matrix
    :param initializer: 'random or zeros' for W
    :param seed: for testing and comparison with deeplearning.ai lecture
    :return: W  and b
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    if initializer == 'random':
        W = np.random.randn(dims[0], dims[1])
    else:
        W = np.zeros(dims)
    b = 0.

    return W, b

def back_prob(A, Y, X):
    m = X.shape[1]
    dW = 1 / m * np.dot(A-Y, X.T)
    dB = 1 / m * np.sum(A-Y, axis=1, keepdims=True)
    return [dW, dB]

def update_params(W, b, grads, learning_rate=0.01):
    W = W - learning_rate * grads[0]
    b = b - learning_rate * grads[1]
    return W, b

def compute_cost(A, Y):
    m = A.shape[1]
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A), axis=1, keepdims=True)
    return np.squeeze(cost)

def forward_prop(W, A_prev, b):
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
    return A


def run_log_reg(X, Y, learning_rate=0.01, number_of_iterations=100, print_costs=False):
    costs = []
    n, m = X.shape

    W, b = initialize_params((1, n), initializer='zeros')  # Initializing W and b with zeros.

    for i in range(1, number_of_iterations + 1):
        A = forward_prop(W, X, b)
        grads = back_prob(A, Y, X)
        cost = compute_cost(A, Y)
        costs.append((i, cost))
        W, b = update_params(W, b, grads, learning_rate)


        if print_costs:
            if i % 100 == 0:
                print('Iteration {}, cost {}'.format(i, cost))

    print("Completed!!!")


    return W, b

def log_reg_predict(X, W, b):
    y_prediction = np.zeros((1, X.shape[1]))
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    y_prediction = 1.0 * (A > 0.5)
    return y_prediction


