import tensorflow as tf
import numpy as np
from utils import binarize
from BinaryLayer import *

BN_EPSILON = 1e-4

class Model(tf.keras.Model):
    def __init__(self,
                 n_classes,
                 n_features,
                 n_hidden_units= 50,
                 learning_rate= 0.1,
                 epochs = 10,
                 decay = 0.9,
                 batch_size = 30,
                 name = 'BinaryLayer',
                 **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.l_1 = BinaryLayer(num_outputs=86)
        self.l_2 = BinaryLayer(num_outputs=64)
        self.l_3 = BinaryLayer(num_outputs=16)

        self.norm_1 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=0.9, epsilon=BN_EPSILON)
        self.norm_2 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=0.9, epsilon=BN_EPSILON)
        self.norm_3 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=0.9, epsilon=BN_EPSILON)

        self.l_output = BinaryLayer(num_outputs=2)

    def forward(self, inputs, in_training_mode, binary=True):
        hidden = self.l_1(inputs, is_binary=True, is_first=True)
        hidden = self.norm_1(hidden)

        hidden = self.l_2(hidden, is_binary=binary)
        hidden = self.norm_2(hidden)


        hidden = self.l_3(hidden, is_binary=binary)
        hidden = self.norm_3(hidden)


        hidden = self.l_output(hidden, is_binary=True, is_first=True)
        #out = binary_tanh_unit(hidden)
        out = tf.nn.log_softmax(hidden)
        #out = hidden
        out = tf.cast(out, dtype=tf.float32)
        return out

    def __call__(self, inputs,  *args, **kwargs):
        self.forward(inputs, in_training_mode=True, binary=False)

    def predictions(self, inputs):
        logits = self.forward(inputs, in_training_mode=False)
        return tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)

