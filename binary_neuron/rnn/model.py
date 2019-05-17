import tensorflow as tf

from binary_neuron.layers import BinaryLinearLayer, HardTanH, SoftMax


def RNNModel(look_back=1):
    model = tf.keras.Sequential([
        # tf.keras.layers.LSTM(4, input_shape=(1, look_back)),
        tf.keras.layers.RNN(cell=tf.keras.layers.LSTMCell(units=4, input_shape=(1, look_back))),
        tf.keras.layers.Dense(1)
    ])
    return model
