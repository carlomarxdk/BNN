import tensorflow as tf


def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2, 0, 1)


def binary_tanh_unit(x):
    return 2 * hard_sigmoid(x) - 1

@tf.custom_gradient
def binarize(x):
    def grad(dy):
        return tf.clip_by_value(tf.identity(dy), -1, 1)
    return tf.sign(x), grad

def data(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((features), labels))
    dataset = dataset.batch(batch_size)
    return dataset

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x,y):
    y_ = tf.transpose(model.forward(x, in_training_mode=False))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))
    return loss


def accuracy(model, input, target):
    with tf.variable_scope('performance'):
        # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
        correct_prediction = tf.equal(tf.cast(tf.argmax(tf.transpose(model.forward(input, in_training_mode=False))),dtype=tf.int32), target)
        ##print(correct_prediction)
        # averaging the one-hot encoded vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
