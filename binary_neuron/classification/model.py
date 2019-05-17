import tensorflow as tf

from binary_neuron.layers import BinaryLinearLayer, HardTanH, SoftMax


def ClassificationModel(classes=2):
    model = tf.keras.Sequential([
        BinaryLinearLayer(50, binarize_input=False),
        HardTanH(),
        BinaryLinearLayer(50),
        HardTanH(),
        BinaryLinearLayer(50),
        HardTanH(),
        BinaryLinearLayer(classes),
        SoftMax()
    ])
    return model