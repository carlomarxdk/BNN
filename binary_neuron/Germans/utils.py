import tensorflow as tf

@tf.custom_gradient
def binarize(x):
    def grad(dy):
        dy = tf.clip_by_value(dy, -1, 1)
        return dy
        #return dy
    return tf.sign(x), grad
    ##return x, grad

@tf.custom_gradient
def round(x):
    def grad(dy):
        #return tf.clip_by_value(dy, -1, 1)
        return dy
    return tf.round(x), grad


def data(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    return dataset