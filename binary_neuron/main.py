import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
from binary_neuron.data_loader import generate_data
from binary_neuron.model import Model
from binary_neuron.train import train

dataset, [features, labels, sizes] = generate_data(num_samples=15000, batch_size=10)

model = Model(classes=2)
optimizer = tf.train.AdamOptimizer(1e-4)
train(model, dataset, optimizer, epochs=100, print_freq=10, sizes=sizes)

# Plotting

colors = {0: '#de0000', 1: '#1f2b7b', 2: '#cfde3c'}
_labels = [colors[label] for label in labels.numpy()]
plt.scatter(features[:, 0], features[:, 1], s=40, c=_labels, cmap=plt.cm.Spectral)
plt.show()

preds = np.argmax(model(features).numpy(), axis=1)
_preds = [colors[pred] for pred in preds]
plt.scatter(features[:, 0], features[:, 1], s=40, c=_preds, cmap=plt.cm.Spectral)
plt.show()

accuracy = round(((1. - (np.sum(np.not_equal(preds, labels.numpy())) / len(preds))) * 100), 2)
print(f'Accuracy: {accuracy}%')
