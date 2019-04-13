import tensorflow as tf
from binary_neuron.utils import binarize, hard_tanh
import numpy as np


class Model(object):
    def __init__(self, classes=2):
        self.weights = [
            tf.get_variable('weight', [200, 5], initializer=tf.contrib.layers.xavier_initializer()),
            tf.get_variable('weight', [200, 200], initializer=tf.contrib.layers.xavier_initializer()),
            tf.get_variable('weight', [200, 200], initializer=tf.contrib.layers.xavier_initializer()),
            tf.get_variable('weight', [200, 200], initializer=tf.contrib.layers.xavier_initializer()),
            tf.get_variable('weight', [classes, 200], initializer=tf.contrib.layers.xavier_initializer())
        ]

    def params(self):
        return self.weights

    def __call__(self, x):
        # enriching data
        _x = x.numpy()
        _x = np.asarray([_x[:, 0], _x[:, 1], _x[:, 0] * _x[:, 1], np.sin(_x[:, 0]), np.sin(_x[:, 1])])
        x = tf.convert_to_tensor(_x, dtype=tf.float32)

        for weight in self.weights[:-1]:
            weight = binarize(weight)
            x = tf.linalg.matmul(weight, x)
            x = hard_tanh(x)

        last_weight = self.weights[-1]
        last_weight = binarize(last_weight)

        x = tf.linalg.matmul(last_weight, x)
        out = tf.nn.softmax(tf.transpose(x))
        return out


if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph = tf.Graph()
    # sess = tf.Session()  # graph=graph)
    # with sess.as_default():
    model = Model()
    model.update([tf.random.uniform([10, 2])], 1)
    print('done')
