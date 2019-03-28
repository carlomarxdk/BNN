import tensorflow as tf
from binary_neuron.utils import binarize, round
import numpy as np


class Model(object):
    def __init__(self, binary=True):
        self.binary = binary
        self.weights = [
            tf.Variable(tf.random.uniform([20, 2], minval=-1.0, maxval=1.0, dtype=tf.float32)),
            tf.Variable(tf.random.uniform([10, 20], minval=-1.0, maxval=1.0, dtype=tf.float32)),
            tf.Variable(tf.random.uniform([2, 10], minval=-1.0, maxval=1.0, dtype=tf.float32))
        ]

    def params(self):
        return self.weights

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            if self.binary:
                weight.assign_sub(tf.clip_by_value(gradients[idx], -0.9, 0.9) * learning_rate)
            else:
                weight.assign_sub(gradients[idx] * learning_rate)

    def __call__(self, x):
        x = tf.convert_to_tensor(x[np.newaxis].T, dtype=tf.float32)

        if self.binary:
            for weight in self.weights[:-1]:
                weight = binarize(weight)
                x = tf.linalg.matmul(weight, x)
                x = binarize(x)

            last_weight = self.weights[-1]
            last_weight = binarize(last_weight)

            x = tf.linalg.matmul(last_weight, x)

        else:
            for weight in self.weights[:-1]:
                x = tf.nn.tanh(tf.linalg.matmul(weight, x))

            last_weight = self.weights[-1]
            x = tf.linalg.matmul(last_weight, x)

        out = tf.sigmoid(x)
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
