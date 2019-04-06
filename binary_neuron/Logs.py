import numpy as np
import tensorflow as tf
from utils import *


def log_prediction(model):
    grid = np.asarray([(i/40 , j/40 ) for j in range(-50, 50) for i in range(-50, 50)])
    x = tf.argmax(tf.transpose(model.forward(tf.Variable(grid, dtype=tf.float32), in_training_mode=False)))
    image = tf.cast(tf.reshape(tf.reshape(x, [-1]), [-1, 100, 100, 1]), dtype=tf.float32)
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.image('Boundry', image)

def log_loss(loss):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('Cross_Entropy_Loss_Per_Epoch', loss)


def log_accuracy(accuracy):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('Accuracy_Per_Epoch', accuracy)

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



def print_loss(losses, epoch, current_loss, current_accuracy):
    print('Epoch %2d: loss=%2.5f accuracy=%2.5f' %
          (epoch, np.asarray(current_loss), np.asarray(current_accuracy)))


