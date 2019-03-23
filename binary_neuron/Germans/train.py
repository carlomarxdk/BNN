import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

## for tensorboard
summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis=10000)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()


def loss(output, target):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    return tf.reduce_mean(tf.square(output - target))

def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape() as t:
        loss_value = loss(model(inputs), outputs)

    gradients = t.gradient(loss_value, model.params())
    model.update(gradients, learning_rate)

def train(model, inputs, targets):

    losses = np.zeros(model.epochs)
    for epoch in range(model.epochs):
        current_loss = []
        for idx, input in enumerate(inputs):
            target = targets[idx]
            current_loss.append(loss(model(input), target))
            backward(model, input, target, loss, learning_rate=model.learning_rate)

        global_step.assign_add(1)
        log_loss(np.asarray(current_loss).mean())
        log_weight(model.params())
        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
        losses[epoch] = np.asarray(current_loss).mean()


def log_loss(loss):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('Loss_Per_Epoch', loss)

def log_weight(w):
    with tf.contrib.summary.always_record_summaries():
        with tf.name_scope('Weights'):
            tf.contrib.summary.histogram('W1', w[0])
            tf.contrib.summary.histogram('W2', w[1])
            tf.contrib.summary.histogram('W3', w[2])





