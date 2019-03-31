import tensorflow as tf
import numpy as np
from utils import binarize
from Logs import *

class Model(object):

    def __init__(self, n_classes, n_features, n_hidden_units=50, learning_rate=0.01, epochs = 100, decay = 0.9):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.weights = self._init_weights()
        self.batch_size = 10

    def _init_weights(self):
        weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
        with tf.variable_scope('Layer1'):
            w1 = tf.get_variable('W1', shape=[self.n_hidden_units*5, self.n_features], initializer=weight_initializer)
        with tf.variable_scope('Layer2'):
            w2 = tf.get_variable('W2', shape=[self.n_hidden_units*10, self.n_hidden_units*5], initializer=weight_initializer)
        with tf.variable_scope('Layer3'):
            w3 = tf.get_variable('W3', shape=[self.n_hidden_units*3, self.n_hidden_units*10], initializer=weight_initializer)
        with tf.variable_scope('Layer4'):
            w4 = tf.get_variable('W4', shape=[self.n_classes, self.n_hidden_units*3], initializer=weight_initializer)

        return [w1, w2, w3, w4]

    def params(self):
        return self.weights

    def forward(self, input):
        self.__call__(self, input)

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            tf.assign_sub(weight, (gradients[idx] * learning_rate))
        ##self.clip()

    def update_learning_rate(self):
        self.learning_rate = self.learning_rate * self.decay
    def clip(self):
        for idx, weight in enumerate(self.weights):
            tf.assign(weight, tf.clip_by_value(weight, -1, 1))
        pass

    def __call__(self, x , training=True):
        ## Get the input
        a = tf.convert_to_tensor(tf.transpose(x), dtype=tf.float32)
        for weight in self.weights[:-1]:
            weight = binarize(weight) ##binirize weights
            a = binarize(a)
            s = tf.matmul(weight, a)
            a = tf.contrib.layers.batch_norm(s, epsilon= 1e-7, decay=self.decay, center=False, scale=False, is_training=training)


        last_weight = self.weights[-1]
        out = tf.matmul(last_weight, a)
        out = tf.nn.log_softmax(out)
        out = tf.cast(out, dtype=tf.float32)
        return out

