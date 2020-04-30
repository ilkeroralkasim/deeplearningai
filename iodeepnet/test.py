import deeplearning
import numpy as np

model = deeplearning.BasicRNN(200)
np.random.seed(1)
model.train(np.random.randn(3, 10, 4), np.random.randn(3, 10, 4), seed=1)
