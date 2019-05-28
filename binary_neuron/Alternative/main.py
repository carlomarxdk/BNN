import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.enable_eager_execution()

import sklearn
import numpy as np
from binary_neuron.train import *
from binary_neuron.model import *

from DataGenerator import *
from sklearn.preprocessing import PolynomialFeatures

data = DataGenerator(num_samples=2000, noise=0.2)
[X_train, y_train] = data.train(num_observations=1000)
[X_test, y_test] = data.test(num_observations=1000)


num_features = data.num_features()

model = Model(n_classes=2, n_features=num_features, learning_rate=1e-5, epochs=1000, decay=0.99)
train(model, X_train, y_train, X_test, y_test)


print(model.predictions(X_train[1,:].reshape(1,-1)), y_train[1])

print(model.predictions(X_train[0,:].reshape(1,-1)), y_train[0])