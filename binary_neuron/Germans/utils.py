import tensorflow as tf

@tf.custom_gradient
def binarize_(x):
    def grad(dy):
        dy = binary_tanh_unit(tf.identity(dy))
        return dy
    return tf.sign(x), grad

@tf.custom_gradient
def binarize(x):
    def grad(dy):
        #dy = binary_tanh_unit(tf.identity(dy))
        return dy
    return tf.sign(binary_tanh_unit(x)), grad

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2, 0, 1)

def binary_tanh_unit(x):
    return 2 * hard_sigmoid(x) - 1

def round(x):
    return tf.round(x)

def data(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((features), labels))
    dataset = dataset.batch(batch_size)
    return dataset

def loss(model, x,y):
    y_ = model.forward(x, training=False)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))
    return loss

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

