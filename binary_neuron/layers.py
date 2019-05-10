import tensorflow as tf

from binary_neuron.utils import binarize, hard_tanh


class BinaryLinearLayer(tf.keras.layers.Layer):
    def __init__(self, num_out, binarize_input=True):
        super(BinaryLinearLayer, self).__init__()
        self.num_out = num_out
        self.binarize_input = binarize_input

    def build(self, input_shape):
        self.weight = tf.get_variable('weight', [int(input_shape[-1]), self.num_out],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    def call(self, x):
        if self.binarize_input:
            x = binarize(x)
        weight = binarize(self.weight)
        return tf.matmul(x, weight)


class HardTanH(tf.keras.layers.Layer):
    def call(self, x):
        return hard_tanh(x)


class SoftMax(tf.keras.layers.Layer):
    def call(self, x):
        return tf.nn.softmax(x)
