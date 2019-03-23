import tensorflow as tf

@tf.custom_gradient
def binarize(x):
    def grad(dy):
        return tf.clip_by_value(dy, -1, 1)
    return tf.sign(x), grad

@tf.custom_gradient
def round(x):
    def grad(dy):
        return tf.clip_by_value(dy, -1, 1)
    return tf.round(x), grad