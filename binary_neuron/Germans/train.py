import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from binary_neuron.Germans.utils import binarize
from binary_neuron.Germans.Logs import *

tf.enable_eager_execution()

## for tensorboard
summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis=10000)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()


def loss_(output, target):
    with tf.variable_scope('loss'):
        target = tf.convert_to_tensor(target, dtype=tf.float32)
    return tf.reduce_mean(tf.square(output - target))

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=tf.transpose(y_))


def CrossEntropy(yHat, y):
    y = tf.cast(y, dtype=tf.float32)
    with tf.variable_scope('loss'):
        # computing cross entropy per sample
        cross_entropy = -tf.reduce_sum(y * tf.log(yHat + 1e-6))
        # Average over samples
        # Averaging makes the loss invariant to batch size, which is very nice.
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def Hinge(yHat, y):
    return tf.maximum(0, 1 - yHat * y)

def accuracy(model, input, target):
    with tf.variable_scope('performance'):
        # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
        correct_prediction = tf.equal(tf.cast(tf.argmax(model(input)),dtype=tf.int32), target)

        # averaging the one-hot encoded vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape() as t:
        loss_value = loss(model, inputs, outputs)

    gradients = t.gradient(loss_value, model.params())
    model.update(gradients, learning_rate)


def train(model, inputs, targets):
    dataset = data(inputs, targets, model.batch_size)
    losses = np.zeros(model.epochs)
    accuracy_est = np.zeros(model.epochs)
    for epoch in range(model.epochs):
        current_loss = np.zeros(model.batch_size)
        current_accuracy = np.zeros(model.batch_size)
        for batch, (X, y) in enumerate(dataset):
            if batch < model.batch_size:
                backward(model, X, y, loss, learning_rate=model.learning_rate)
                current_loss[batch] = (loss(model,X, y)).numpy()
                current_accuracy[batch] = accuracy(model,X,y)

        losses[epoch] = np.asarray(current_loss).mean()
        accuracy_est[epoch] = np.asarray(current_accuracy).mean()
        print_loss(losses, epoch, losses[epoch], accuracy_est[epoch])

        model.update_learning_rate()
        global_step.assign_add(1)
        log_loss(np.asarray(losses[epoch]))
        log_weight(model.params())
        log_prediction(model)


