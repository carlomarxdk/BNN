import numpy as np
import tensorflow as tf


@tf.custom_gradient
def hard_tanh(x):
    def grad(dy):
        return dy

    return tf.clip_by_value(x, -1., 1.), grad


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


def enrich_data(features):
    return np.asarray([features[:, 0],
                       features[:, 1],
                       features[:, 0] * features[:, 1],
                       np.sin(features[:, 0]),
                       np.cos(features[:, 1]),
                       np.square(features[:, 0]),
                       np.square(features[:, 1])
                       ]).T
