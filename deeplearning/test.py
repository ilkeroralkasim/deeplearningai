import deeplearning
import numpy as np

model = deeplearning.BasicRNN(200)
x_train = np.random.randn(10, 5, 3)
y_train = np.random.randn(10, 5, 3)

model.train(x_train, y_train)
