import tensorflow as tf
from binary_neuron.utils import binarize, round
import numpy as np


class Model(object):
    def __init__(self):
        self.weights = [
            tf.Variable(tf.random.uniform([100, 2], minval=-1.0, maxval=1.0, dtype=tf.float32)),
            tf.Variable(tf.random.uniform([50, 100], minval=-1.0, maxval=1.0, dtype=tf.float32)),
            tf.Variable(tf.random.uniform([1, 50], minval=-1.0, maxval=1.0, dtype=tf.float32))
        ]

    def params(self):
        return self.weights

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            weight.assign_sub(tf.clip_by_value(gradients[idx], -1.0, 1.0) * learning_rate)

    def __call__(self, x):
        x = tf.convert_to_tensor(x[np.newaxis].T, dtype=tf.float32)

        for weight in self.weights[:-1]:
            weight = binarize(weight)
            x = tf.linalg.matmul(weight, x)
            x = binarize(x)

        last_weight = self.weights[-1]
        last_weight = binarize(last_weight)

        out = tf.linalg.matmul(last_weight, x)
        out = tf.sigmoid(out)
        out = round(out)
        return tf.reshape(out, [-1])


if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph = tf.Graph()
    # sess = tf.Session()  # graph=graph)
    # with sess.as_default():
    model = Model()
    model.update([tf.random.uniform([10, 2])], 1)
    print('done')
