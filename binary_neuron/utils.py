import tensorflow as tf


@tf.custom_gradient
def binarize(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


@tf.custom_gradient
def round(x):
    def grad(dy):
        return dy

    return tf.round(x), grad
