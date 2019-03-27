import tensorflow as tf
import numpy as np
from binary_neuron.Germans.utils import binarize, round
from binary_neuron.Germans.Logs import *

class Model(object):

    def __init__(self, n_classes, n_features, n_hidden_units=10, learning_rate=0.01, epochs = 50, decay = 0.9 ):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.weights = self._init_weights()
        self.batch_size = 100

    def _init_weights(self):
        weight_initializer = tf.truncated_normal_initializer(stddev=10)
        with tf.variable_scope('Layer1'):
            w1 = tf.get_variable('W1', shape=[self.n_hidden_units, self.n_features], initializer=weight_initializer)
        with tf.variable_scope('Layer2'):
            w2 = tf.get_variable('W2', shape=[self.n_hidden_units*2, self.n_hidden_units], initializer=weight_initializer)
        with tf.variable_scope('Layer3'):
            w3 = tf.get_variable('W3', shape=[self.n_classes, self.n_hidden_units*2], initializer=weight_initializer)

        return [w1, w2, w3]

    def params(self):
        return self.weights

    def forward(self, input):
        self.__call__(self, input)

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            tf.assign_sub(weight, (gradients[idx] * learning_rate))

    def update_learning_rate(self):
        self.learning_rate = self.learning_rate * self.decay

    def __call__(self, x):
        ## Get the input
        a = tf.convert_to_tensor(tf.transpose(x), dtype=tf.float32)
        for weight in self.weights[:-1]:
            weight = binarize(weight) ##binirize weights
            a = tf.matmul(weight, a)
            a = binarize(a) ## binirize input of the previous layer

        last_weight = binarize(self.weights[-1])
        out = tf.matmul(last_weight, a)
        out = tf.nn.log_softmax(out)
        out = tf.cast(out, dtype=tf.float32)
        return out

