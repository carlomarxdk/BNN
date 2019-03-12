import tensorflow as tf
from binary_neuron.utils import binarize, hard_tanh


class Model(object):
    def __init__(self):
        self.weights = [
            tf.Variable(tf.random.uniform([20, 2], dtype=tf.float32,
                                          minval=tf.constant(-1, dtype=tf.float32))),
            tf.Variable(tf.random.uniform([1, 20], dtype=tf.float32,
                                          minval=tf.constant(-1, dtype=tf.float32)))
        ]

    def params(self):
        return self.weights

    def update(self, gradients, learning_rate):
        for idx, weight in enumerate(self.weights):
            weight.assign_sub(gradients[idx] * learning_rate)

    def __call__(self, x):
        for weight in self.weights[:-1]:
            weight = binarize(weight)
            x = tf.linalg.matmul(weight, x)
            x = binarize(x)

        last_weight = self.weights[-1]
        # last_weight = binarize(last_weight)

        out = tf.linalg.matmul(last_weight, x)
        return tf.reshape(out, [-1])


if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph = tf.Graph()
    # sess = tf.Session()  # graph=graph)
    # with sess.as_default():
    model = Model()
    model.update([tf.random.uniform([10, 2])], 1)
    print('done')
