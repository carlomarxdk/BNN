import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.enable_eager_execution()
from binary_neuron.data_loader import generate_data
from binary_neuron.model import Model
from binary_neuron.train import train

dataset, [features, labels] = generate_data(15000, 20)

model = Model()
optimizer = tf.train.AdamOptimizer(1e-5)
train(model, dataset, optimizer, epochs=200, print_freq=20)

colors = {0: '#de0000', 1: '#1f2b7b'}
preds = np.argmax(model(tf.convert_to_tensor(features)).numpy(), axis=1)
labels = [colors[pred] for pred in preds]
plt.scatter(features[:, 0], features[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
plt.show()
