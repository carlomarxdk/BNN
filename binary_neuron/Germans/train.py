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


def loss(output, target):
    with tf.variable_scope('loss'):
        target = tf.convert_to_tensor(target, dtype=tf.float32)
    return tf.reduce_mean(tf.square(output - target))

def CrossEntropy(yHat, y):
    y = tf.cast(y, dtype=tf.float32)
    with tf.variable_scope('loss'):
        # computing cross entropy per sample
        ##print(yHat,y)

        cross_entropy = -tf.reduce_sum(y * tf.log(yHat + 1e-6), reduction_indices=[1])
        # Average over samples
        # Averaging makes the loss invariant to batch size, which is very nice.
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy

def loss_fn(model, x, y):
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=model(x), labels=y))

def Hinge(yHat, y):
    return tf.maximum(0, 1 - yHat * y)

def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape() as t:
        loss_value = loss(model, inputs, outputs)

    gradients = t.gradient(loss_value, model.params())
    print(gradients, loss_value)
    model.update(gradients, learning_rate)

def train(model, inputs, targets):
    dataset = data(inputs, targets, model.batch_size)
    losses = np.zeros(model.epochs)
    for epoch in range(model.epochs):
        current_loss = np.zeros(model.batch_size)
        for batch, (X, y) in enumerate(dataset):
            if batch < model.batch_size:
                backward(model, X, y, CrossEntropy, learning_rate=model.learning_rate)
                current_loss[batch]= (CrossEntropy(model, X, y)).numpy()

        losses[epoch] = np.asarray(current_loss).mean()
        print_loss(losses, epoch, losses[epoch])

        model.update_learning_rate()
        global_step.assign_add(1)
        ##print(current_loss)
        log_loss(np.asarray(losses[epoch]))
        log_weight(model.params())
        #log_prediction(model)




