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
X_train = data.train[0][:5000]
y_train = data.train[1][:5000]

model = Model(n_classes=10, learning_rate=1e-5, epochs=50, decay=0.99)
train(model, X_train, y_train)
