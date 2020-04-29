import numpy as np

class DeepNN:
  def __init__(self, layers=[]):
    self.layers = layers

  def train(self, x_train, y_train):
    return 0


def tanh(Z):
  return np.tanh(Z)


def relu(Z):
  return np.max(0, Z)


def sigmoid(Z):
  return 1 / (1 + np.exp(-Z))


