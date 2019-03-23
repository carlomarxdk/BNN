import tensorflow as tf
import numpy as np
from binary_neuron.Germans.utils import binarize

class Model(object):

    def __init__(self, n_classes, n_features, n_hidden_units=10, learning_rate=0.01, epochs = 10, ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = self._init_weights()

    def _init_weights(self):
        w1 = tf.Variable(tf.random.uniform(shape=(self.n_hidden_units, self.n_features), minval=-1.0, maxval=1.0))
        w2 = tf.Variable(tf.random.uniform(shape=(self.n_hidden_units*10, self.n_hidden_units), minval=-1.0, maxval=1.0))
        w3 = tf.Variable(tf.random.uniform(shape=(self.n_hidden_units*20, self.n_hidden_units*10), minval=-1.0, maxval=1.0))
        w4 = tf.Variable(tf.random.uniform(shape=(1, self.n_hidden_units*20), minval=-1.0, maxval=1.0))

        return [w1, w2, w3 , w4]

    def params(self):
        return self.weights

    def forward(self, input):
        self.__call__(self, input)

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            tf.assign_sub(weight, (gradients[idx] * learning_rate))

    def __call__(self, x):
        ## Get the input
        x = tf.convert_to_tensor(x[np.newaxis].T, dtype=tf.float32)
        a = binarize(x)
        for weight in self.weights[:-1]:
            weight = binarize(weight) ##binirize weights
            a = tf.tensordot(weight, a, axes=1)
            a = binarize(a) ## binirize input of the previous layer

        last_weight = binarize(self.weights[-1])
        out = tf.linalg.matmul(last_weight, a)
        out = tf.sigmoid(out)
        return tf.reshape(out, [-1])

