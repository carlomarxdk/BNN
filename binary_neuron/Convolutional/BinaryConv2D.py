import tensorflow as tf
from utils import *

class BinaryConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters= 16, kernel_size = (3,3), padding = 'same'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        super(BinaryConv2D, self).__init__(filters=self.filters, kernel_size=self.kernel_size,
                                           use_bias=False, padding=self.padding)

    def call(self, input, is_binary = True):
        weight = binarize(self.kernel, binary=True)
        outputs = self._convolution_op(input, weight)

        if is_binary:
            outputs = binarize(outputs)

        return outputs

