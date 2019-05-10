import tensorflow as tf

from binary_neuron.utils import binarize, hard_tanh


def binaryLinearLayer(num_out, name=None):
    def binary_linear_layer(x):
        with tf.variable_scope(name, 'BinaryLinear', [x]):
            num_in = x.get_shape().as_list()[0]
            x_binary = binarize(x)
            w = tf.get_variable('weight', [num_out, num_in],
                                initializer=tf.contrib.layers.xavier_initializer())
            # w = hard_tanh(w)
            w_binary = binarize(w)
            out = tf.matmul(w_binary, x_binary)
        return out

    return binary_linear_layer


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
