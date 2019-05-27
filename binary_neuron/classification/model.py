import tensorflow as tf

from binary_neuron.layers import BinaryLinearLayer, HardTanH, SoftMax, Sign


def ClassificationModel(classes=2):
    model = tf.keras.Sequential([
        BinaryLinearLayer(10, binarize_input=False),
        Sign(),
        BinaryLinearLayer(classes),
        SoftMax()
    ])
    return model


def BaselineClassificationModel(classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_dim=7),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(classes),
        tf.keras.layers.Softmax()
    ])
    return model
