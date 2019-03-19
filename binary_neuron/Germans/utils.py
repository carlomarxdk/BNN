import tensorflow as tf

@tf.custom_gradient
def binarize_(tensor):
    ones = tf.ones(tensor.shape, tf.float32)
    twos = tf.cast(tf.fill(tensor.shape, 2), tf.float32)

    tensor = tf.greater_equal(tensor, 0)  # True, False
    tensor = tf.cast(tensor, tf.float32)  # 0, 1
    tensor = tf.multiply(tensor, twos)  # 0, 2
    return tf.subtract(tensor, ones)  # -1, 1

@tf.custom_gradient
def binarize(x):
    def grad(dy):
        return tf.clip_by_value(dy, -1.0, 1.0)

    return tf.sign(x), grad


def batch_norm(inputs, weights):
    mean, variance = tf.nn.moments(inputs, axis=0)
    x_hat = tf.math.add(inputs, -mean)/ tf.math.sqrt(variance) ## we can add noise
    return x_hat

def mini_batch(inputs, outputs, n_batches = 10):
    return [tf.split(inputs, n_batches, axis=0) ,
           tf.split(outputs, n_batches, axis=0)]