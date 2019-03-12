import tensorflow as tf
import numpy as np
from binary_neuron.utils import clip


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


def backward(model, inputs, targets, loss, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), targets)
    # print('Inputs: ', inputs.numpy())
    # print('Target: ', targets.numpy())
    # print('Predictions: ', model(inputs).numpy())
    # print('Losses: ', current_loss)
    # print('Params: ', model.params())

    gradients = t.gradient(current_loss, model.params())

    clipped_gradients = [clip(gradient) for gradient in gradients[:-1]]
    clipped_gradients.append(gradients[-1])

    # print('Gradients: ', gradients)
    model.update(clipped_gradients, learning_rate)


def train(model, inputs, targets, epochs=10):
    for epoch in range(epochs):
        current_loss = []
        for i in range(inputs.shape[0]):
            input = tf.convert_to_tensor(inputs[np.newaxis, i].T, dtype=tf.float32)
            target = tf.convert_to_tensor(targets[i], dtype=tf.float32)
            current_loss.append(loss(model(input), target))

            backward(model, input, target, loss, learning_rate=1e-4)

        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
