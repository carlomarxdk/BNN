import tensorflow as tf
import numpy as np


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape(persistent=True) as t:
        loss_value = loss(model(inputs), outputs)
    gradients = t.gradient(loss_value, model.params())
    print(loss_value, gradients)

    #model.update(gradients, learning_rate)

def train(model, inputs, targets, epochs=10):
    # Collect the history of W-values and b-values to plot later
    Ws, bs = [], []
    for epoch in range(model.epochs):
        current_loss = []
        for i in range(1,inputs.shape[0]):
            input = tf.cast(inputs[i], dtype= tf.float32)
            target = tf.convert_to_tensor(targets[i], dtype=tf.float32)
            Ws.append(model.W1.numpy())
            current_loss.append(loss(model.forward(input), target))

            backward(model, input, target, loss, model.learning_rate)

        print('Epoch %2d: loss=%2.5f' %
            (epoch, np.asarray(current_loss).mean()))

