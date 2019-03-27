import numpy as np
import tensorflow as tf
from binary_neuron.Germans.utils import *


def log_prediction(model):
    grid = np.asarray([(i / 10, j / 10) for j in range(-20, 20) for i in range(-20, 20)])
    x= np.asarray([tf.argmax(model(input)).numpy() for input in grid]).flatten()
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
            #tf.contrib.summary.histogram('Layer_3', w[2])
            #tf.contrib.summary.histogram('Layer_4', w[3])

        with tf.name_scope('Binary Weights'):
            tf.contrib.summary.histogram('Layer_1', binarize(w[0]))
            tf.contrib.summary.histogram('Layer_2', binarize(w[1]))
            #tf.contrib.summary.histogram('Layer_3', binarize(w[2]))
            #tf.contrib.summary.histogram('Layer_4', binarize(w[3]))


def print_loss(losses, epoch, current_loss):
    print('Epoch %2d: loss=%2.5f' %
          (epoch, np.asarray(current_loss).sum()))


