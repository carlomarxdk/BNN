import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.enable_eager_execution()
from binary_neuron.data_loader import generate_data
from binary_neuron.model import Model
from binary_neuron.train import train

dataset, [features, labels, sizes] = generate_data(15000, 20)

model = Model(classes=2)
optimizer = tf.train.AdamOptimizer(1e-4)
train(model, dataset, optimizer, epochs=100, print_freq=10, sizes=sizes)

# Plotting

colors = {0: '#de0000', 1: '#1f2b7b', 2: '#cfde3c'}
labels = [colors[label] for label in labels]
plt.scatter(features[:, 0], features[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
plt.show()

_preds = np.argmax(model(tf.convert_to_tensor(features)).numpy(), axis=1)
preds = [colors[pred] for pred in _preds]
plt.scatter(features[:, 0], features[:, 1], s=40, c=preds, cmap=plt.cm.Spectral)
plt.show()
