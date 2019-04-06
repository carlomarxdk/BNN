import tensorflow as tf
import numpy as np
from utils import binarize
from BinaryLayer import *

class Model(tf.keras.Model):
    def __init__(self,
                 n_classes,
                 n_features,
                 n_hidden_units= 50,
                 learning_rate= 0.01,
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
        self.l_1 = BinaryLayer(num_outputs=10)
        self.l_2 = BinaryLayer(num_outputs=self.n_hidden_units)
        self.l_output = BinaryLayer(num_outputs=2)


    def forward(self, inputs, in_training_mode, binary=True):
        hidden = self.l_1(inputs, binary=binary)
        hidden = tf.keras.layers.BatchNormalization()(hidden, training=in_training_mode)

        hidden = self.l_output(hidden, binary=False)
        out = tf.nn.log_softmax(hidden)
        out = tf.cast(out, dtype=tf.float32)
        return out

    def __call__(self, inputs,  *args, **kwargs):
        self.forward(inputs, in_training_mode=True, binary=False)

    def prediction(self, inputs):
        logits = self.forward(inputs, in_training_mode=False)
        return tf.argmax(logits, axis = 1)

