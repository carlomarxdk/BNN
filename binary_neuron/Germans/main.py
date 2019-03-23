import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

from binary_neuron.Germans.data_loader import data, labels, inputs, outputs
from binary_neuron.Germans.model import Model
from binary_neuron.Germans.train import *

model = Model(n_classes=2, n_features=2, n_hidden_units=5, learning_rate=0.01,  epochs=10)

train(model, data,labels)

