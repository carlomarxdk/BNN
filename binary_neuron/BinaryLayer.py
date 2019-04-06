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

    def call(self, input, binary = True):
        if binary:
            return tf.matmul(binarize(input), binarize(self.kernel))
        return tf.matmul(input, self.kernel)

