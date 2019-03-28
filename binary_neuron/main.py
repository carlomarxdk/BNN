import tensorflow as tf

tf.enable_eager_execution()
from binary_neuron.data_loader import data, labels
from binary_neuron.model import Model
from binary_neuron.train import train

model = Model(binary=True)
train(model, data, labels, epochs=100)

train(model, data, labels, epochs=100)
>>>>>>> parent of 97db9b1... Update
