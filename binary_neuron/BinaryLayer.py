import tensorflow as tf
from utils import *

class BinaryLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout = 1):
        self.num_outputs = num_outputs
        self.dropout=dropout
        super(BinaryLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]), self.num_outputs],
                                        initializer = tf.contrib.layers.xavier_initializer(),
                                        trainable=True)

    def call(self, input, is_binary = True):
        weight = binarize(self.kernel, binary=True)
        output = tf.matmul(input, weight)
        if is_binary:
            output = binarize(output)

        return tf.nn.dropout(output, self.dropout)

