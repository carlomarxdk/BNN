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

data = DataGenerator(num_samples=1500, noise=0.2)
[X_train, y_train] = data.train(num_observations=1000)

num_features = data.num_features()

model = Model(n_classes=2, n_features=num_features, learning_rate=1e-5, epochs=100, decay=0.9)
train(model, X_train, y_train)


print(model.predictions(X_train[1,:].reshape(1,-1)), y_train[1])

print(model.predictions(X_train[0,:].reshape(1,-1)), y_train[0])