import tensorflow as tf
import numpy as np
from utils import binarize

class Model(object):

    def __init__(self, n_classes, n_features, n_hidden_units=10, learning_rate=0.01, n_batches = 1, epochs = 10, random_seed = 12):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_batches = n_batches
        self.random_seed = random_seed
        [self.W1, self.W2] = self._init_weights()

    def _init_weights(self):
        w1 = tf.random.uniform(shape=(self.n_hidden_units, self.n_features), minval=-1.0, maxval=1.0)
        w2 = tf.random.uniform(shape=(self.n_hidden_units, self.n_features), minval=-1.0, maxval=1.0)
        return w1, w2

    def params(self):
        return [self.W1]

    def forward(self, input):
        net_hidden = tf.tensordot(input, self.W1, axes=0)
        act_hidden = tf.sigmoid(net_hidden)
        result = (tf.transpose(tf.reduce_sum(act_hidden,axis=0)))
        #result = tf.reduce_sum(act_hidden,axis=0)
        return (tf.cast(tf.reshape(tf.argmax(result, axis=1), [-1]), dtype= tf.float32))

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.W1):
            tf.assign_sub(weight, (gradients[idx] * learning_rate))


    def __call__(self, input):
        net_hidden = tf.tensordot(input, self.W1, axes=0)
        act_hidden = tf.sigmoid(net_hidden)
        result = (tf.transpose(tf.reduce_sum(act_hidden,axis=0)))
        #result = tf.reduce_sum(act_hidden,axis=0)
        result = tf.cast(tf.argmax(result, axis=1), dtype= tf.float32)
        print(result)
        return result