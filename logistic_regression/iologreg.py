import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from ioreg_funcs import *

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
'''
train_set_x_orig:   training photos, dimension: (209,64,64,3)
train_set_y:        training set labels, dimension: (1, 209)
test_set_x_orig:    test set, dimension: (50, 64, 64, 3)
test_set_y:         test set labels, dimensions:(1, 50)  
classes:            classes, [b'non-cat' b'cat']:(2,)
'''

m = train_set_x_orig.shape[0]  # number of training samples
X_train = train_set_x_orig.reshape(m, -1).T / 255   # X: input vector (64*64*3, 209) and normalized.
Y_train = train_set_y

m = test_set_x_orig.shape[0]
X_test = test_set_x_orig.reshape(m, -1).T / 255
Y_test = test_set_y


learning_rates = [0.1, 0.01, 0.001, 0.0001]

for lr in learning_rates:
    W, b = run_log_reg(X_train, Y_train, learning_rate=lr, number_of_iterations=1500)
    predictions = log_reg_predict(X_test, W, b)
    print('test accuracy: {}%'.format(100 - np.mean(np.abs(predictions-Y_test)) * 100))
    predictions = log_reg_predict(X_train, W, b)
    print('train accuracy: {}%'.format(100 - np.mean(np.abs(predictions - Y_train)) * 100))