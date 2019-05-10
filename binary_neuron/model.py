import tensorflow as tf

from binary_neuron.layers import BinaryLinearLayer, HardTanH, SoftMax
from binary_neuron.utils import binarize, hard_tanh


# class Model(object):
#     def __init__(self, classes=2):
#         self.weights = [
#             tf.get_variable('weight', [7, 200], initializer=tf.contrib.layers.xavier_initializer()),
#             tf.get_variable('weight', [200, 200], initializer=tf.contrib.layers.xavier_initializer()),
#             tf.get_variable('weight', [200, classes], initializer=tf.contrib.layers.xavier_initializer())
#         ]
#
#         self.trainable_variables = self.weights
#
#     def __call__(self, x):
#         for weight in self.weights[:-1]:
#             weight = binarize(weight)
#             x = tf.linalg.matmul(x, weight)
#             x = hard_tanh(x)
#
#         last_weight = binarize(self.weights[-1])
#
#         x = tf.linalg.matmul(x, last_weight)
#
#         out = tf.nn.softmax(x)
#         return out


def Model(classes=2):
    model = tf.keras.Sequential([
        BinaryLinearLayer(50, binarize_input=False),
        HardTanH(),
        BinaryLinearLayer(50),
        HardTanH(),
        BinaryLinearLayer(50),
        HardTanH(),
        BinaryLinearLayer(classes),
        SoftMax()
    ])
    return model


if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph = tf.Graph()
    # sess = tf.Session()  # graph=graph)
    # with sess.as_default():
    model = Model()
    model.update([tf.random.uniform([10, 2])], 1)
    print('done')
