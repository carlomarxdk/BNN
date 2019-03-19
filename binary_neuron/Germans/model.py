import tensorflow as tf
import numpy as np
from binary_neuron.Germans.utils import binarize

class Model(object):

    def __init__(self, n_classes, n_features, n_hidden_units=10, learning_rate=0.01, n_batches = 1, epochs = 10, random_seed = 12):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_batches = n_batches
        self.random_seed = random_seed
        self.weights = self._init_weights()

    def _init_weights(self):
        w1 = tf.Variable(tf.random.uniform(shape=(self.n_hidden_units*2, self.n_features), minval=-1.0, maxval=1.0))
        w2 = tf.Variable(tf.random.uniform(shape=(self.n_hidden_units, self.n_hidden_units*2), minval=-1.0, maxval=1.0))
        w3 = tf.Variable(tf.random.uniform(shape=(1, self.n_hidden_units), minval=-1.0, maxval=1.0))
        return [w1,w2, w3]

    def params(self):
        return self.weights

    def forward(self, input):
        self.__call__(self, input)

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            tf.assign_sub(weight, (gradients[idx] * learning_rate))

    def __call__(self, x):
        x = tf.convert_to_tensor(x[np.newaxis].T, dtype=tf.float32)

        for weight in self.weights[:-1]:
            weight = binarize(weight)
            x = tf.tensordot(weight, x, axes=1)
            x = binarize(x)

        last_weight = self.weights[-1]
        last_weight = binarize(last_weight)

        out = tf.linalg.matmul(last_weight, x)
        out = tf.sigmoid(out)
        return tf.reshape(out, [-1])

