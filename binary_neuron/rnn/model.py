import tensorflow as tf

from binary_neuron.layers import BinaryLinearLayer, HardTanH, SoftMax, BinaryLSTMCell


def RNNModel(look_back=1):
    model = tf.keras.Sequential([
        tf.keras.layers.RNN(cell=BinaryLSTMCell(units=look_back, input_shape=(1, look_back))),
        tf.keras.layers.Dense(1)
    ])
    return model


def BaselineRNNModel(look_back=1):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(4, input_shape=(1, look_back)),
        tf.keras.layers.Dense(1)
    ])
    return model
