import tensorflow as tf


def binarize(tensor):
    ones = tf.ones(tensor.shape, tf.float32)
    twos = tf.cast(tf.fill(tensor.shape, 2), tf.float32)

    tensor = tf.greater_equal(tensor, 0)  # True, False
    tensor = tf.cast(tensor, tf.float32)  # 0, 1
    tensor = tf.multiply(tensor, twos)  # 0, 2
    return tf.subtract(tensor, ones)  # -1, 1
