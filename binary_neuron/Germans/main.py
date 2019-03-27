import tensorflow as tf
import sklearn
import numpy as np
tf.enable_eager_execution()

from binary_neuron.Germans.data_loader import *
from binary_neuron.Germans.model import Model
from binary_neuron.Germans.train import *

num_features = X_train.shape[-1]
num_output = 2

model = Model(n_features=num_features, n_hidden_units=10, n_classes=num_output , learning_rate=0.001)
##model = Model(n_classes=2, n_features=2, n_hidden_units=5, learning_rate=0.05, decay=0.9,  epochs=10)

train(model, X_train,y_train)

