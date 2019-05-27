import tensorflow as tf
import numpy as np
from utils import binarize
from binary_neuron.Convolutional.BinaryConv2D import *
from binary_neuron.BinaryLayer import *

BN_EPSILON = 1e-6


class Model(tf.keras.Model):
    def __init__(self,
                 n_classes,
                 learning_rate=0.1,
                 epochs=10,
                 decay=1,
                 batch_size=30,
                 name='BinaryLayer',
                 **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.l_1 = BinaryConv2D(filters=48, kernel_size=(3, 3), padding='same')
        self.l_2 = BinaryConv2D(filters=48, kernel_size=(3,3), padding='valid')
        self.pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #dropout
        self.l_3 = BinaryConv2D(filters=96, kernel_size=(3,3), padding='same')
        self.l_4 = BinaryConv2D(filters=96, kernel_size=(3,3), padding='valid')
        self.pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.l_5 = BinaryConv2D(filters=192, kernel_size=(3,3), padding='same')
        self.l_6 = BinaryConv2D(filters=192, kernel_size=(3,3), padding='valid')
        self.pool_6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.dense_1 = BinaryLayer(num_outputs=512)
        self.dense_2 = BinaryLayer(num_outputs=256)
        self.l_output = BinaryLayer(num_outputs=n_classes)

    def forward(self, inputs, in_training_mode, binary=True):
        transformed = tf.cast(inputs, dtype=tf.float32)
        hidden = self.l_1(transformed, is_binary=True)
        hidden = self.l_2(hidden, is_binary=True)
        hidden = self.pool_2(hidden)

        hidden = self.l_3(hidden, is_binary=True)
        hidden = self.l_4(hidden, is_binary=True)
        hidden = self.pool_4(hidden)

        hidden = self.l_5(hidden, is_binary=True)
        hidden = self.l_6(hidden, is_binary=True)
        hidden = self.pool_6(hidden)
        flat = tf.reshape(hidden, shape=[-1, hidden.shape[1] * hidden.shape[2] * hidden.shape[3]]) #depends on the size of the hidden tensor

        flat = self.dense_1(flat, is_binary=True)
        flat = self.dense_2(flat, is_binary=True)
        flat = self.l_output(flat, is_binary=False)
        out = tf.nn.softmax(flat)
        out = tf.cast(out, dtype=tf.float32)
        return out

    def __call__(self, inputs, *args, **kwargs):
        self.forward(inputs, in_training_mode=True, binary=True)

    def predictions(self, inputs):
        logits = self.forward(inputs, in_training_mode=False)
        logits = tf.argmax(logits, axis=1)
        return tf.cast(logits, dtype=tf.int32)