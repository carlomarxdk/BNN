import tensorflow as tf
from utils import *

class BinaryLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        super(BinaryLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]), self.num_outputs],
                                        initializer = tf.initializers.random_normal(),
                                        trainable=True)

    def call(self, input, is_binary = True, is_first=False):
        tf.assign(self.kernel, clip_weight(self.kernel))
        if is_first:
            return tf.matmul(binarize(input, binary=False), binarize(self.kernel, binary=is_binary))

        result = tf.matmul(binarize(input, binary=is_binary), binarize(self.kernel, binary=is_binary))
        return tf.nn.dropout(result, 0.7)

