import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.enable_eager_execution()

import sklearn
import numpy as np
from binary_neuron.Convolutional.train import *
from binary_neuron.Convolutional.model import *
from binary_neuron.Convolutional.DataGenerator import *

data = DataGenerator()
X_train = data.train[0][:11000]
y_train = data.train[1][:11000]
X_test = data.test[0][:1000]
y_test = data.test[1][:1000]

model = Model(n_classes=10, learning_rate=1e-6, epochs=1000, decay=0.9999)
train(model, X_train, y_train, X_test, y_test)
