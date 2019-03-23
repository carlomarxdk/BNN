import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from binary_neuron.Germans.utils import binarize
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
    _gradients = []
    for gradient in gradients:
        max = tf.norm(gradient)
        if not tf.equal(max, 0):
            _gradients.append(gradient / max)
        else:
            _gradients.append(gradient)
    gradients = _gradients

    model.update(gradients, learning_rate)

def train(model, inputs, targets):
    losses = np.zeros(model.epochs) #vector of losses per epoch
    for epoch in range(model.epochs):
        current_loss = []
        for idx, input in enumerate(inputs):
            target = targets[idx]
            current_loss.append(loss(model(input), target))
            backward(model, input, target, loss, learning_rate=model.learning_rate)

        global_step.assign_add(1)
        log_loss(np.asarray(current_loss).mean())
        log_weight(model.params())
        log_prediction(model)
        print_loss(losses, epoch, current_loss)

## IMAGE!!!!
def log_prediction(model):
    grid = np.asarray([(i / 10, j / 10) for j in range(-20, 20) for i in range(-20, 20)])
    x= np.asarray([model(input).numpy() for input in grid]).flatten()
    print(x.shape)
    image = tf.reshape(tf.reshape(x, [-1]), [-1, 40, 40, 1])

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.image('Boundry', image)

def log_loss(loss):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('Loss_Per_Epoch', loss)

def log_weight(w):
    with tf.contrib.summary.always_record_summaries():
        with tf.name_scope('Weights'):
            tf.contrib.summary.histogram('Layer_1', w[0])
            tf.contrib.summary.histogram('Layer_2', w[1])
            tf.contrib.summary.histogram('Layer_3', w[2])
            tf.contrib.summary.histogram('Layer_4', w[3])

        with tf.name_scope('Binary Weights'):
            tf.contrib.summary.histogram('Layer_1', binarize(w[0]))
            tf.contrib.summary.histogram('Layer_2', binarize(w[1]))
            tf.contrib.summary.histogram('Layer_3', binarize(w[2]))
            tf.contrib.summary.histogram('Layer_4', binarize(w[3]))


def print_loss(losses, epoch, current_loss):
    print('Epoch %2d: loss=%2.5f' %
          (epoch, np.asarray(current_loss).mean()))
    losses[epoch] = np.asarray(current_loss).mean()





