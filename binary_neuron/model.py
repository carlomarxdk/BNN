import tensorflow as tf
import numpy as np
from utils import binarize
from binary_neuron.BinaryLayer import *
from sklearn.preprocessing import PolynomialFeatures

BN_EPSILON = 1e-6

class Model(tf.keras.Model):
    def __init__(self,
                 n_classes,
                 transformer,
                 n_features = 2,
                 n_hidden_units= 50,
                 learning_rate= 0.1,
                 epochs = 10,
                 decay = 0.9,
                 batch_size = 30,
                 name = 'BinaryLayer',
                 **kwargs):
        super(Model, self).__init__(name=name, **kwargs)
        self.transformer = transformer
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.l_1 = BinaryLayer(num_outputs=11)
        self.l_2 = BinaryLayer(num_outputs=100)
        self.l_3 = BinaryLayer(num_outputs=82)
        self.l_4 = BinaryLayer(num_outputs=100)
        self.l_5 = BinaryLayer(num_outputs=4)

        #self.norm_1 = tf.keras.layers.BatchNormalization(center=True, scale=True, momentum=0.7, epsilon=BN_EPSILON)
        #self.norm_2 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=0.999, epsilon=BN_EPSILON)
        #self.norm_3 = tf.keras.layers.BatchNormalization(center=False, scale=False, momentum=0.999, epsilon=BN_EPSILON)

        self.l_output = BinaryLayer(num_outputs=1)

    def forward(self, inputs, in_training_mode, binary=True, to_print=False):

        transformed = self.transformer.transform(inputs)
       # if to_print: print('TR: ', transformed)
        transformed = tf.convert_to_tensor(transformed, dtype=tf.float32)
        hidden = self.l_1(transformed, is_binary=True)
        hidden = self.l_2(hidden, is_binary=True)
        hidden = self.l_3(hidden, is_binary=True)
        hidden = self.l_4(hidden, is_binary=True)
        hidden = self.l_5(hidden, is_binary=True)

        hidden = self.l_output(hidden, is_binary=False)
        out = tf.round(hard_sigmoid(hidden))
        if to_print: print('OUT:', out)
        out = tf.cast(hidden, dtype=tf.float32)
        return out

    def __call__(self, inputs,  *args, **kwargs):
        self.forward(inputs, in_training_mode=True, binary=False)

    def predictions(self, inputs, to_print=False):
        logits = self.forward(inputs, in_training_mode=False, to_print=to_print)
        return tf.cast(logits, dtype=tf.int32)

