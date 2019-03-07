import tensorflow as tf
import numpy as np


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    gradients = t.gradient(current_loss, model.params())
    # model.W.assign_sub(learning_rate * dW)
    # model.b.assign_sub(learning_rate * db)
    
    gradients[inputs> 1] = 0
    gradients[inputs<-1] = 0
    model.update(gradients, learning_rate)


def train(model, inputs, targets, epochs=10):
    # Collect the history of W-values and b-values to plot later
    Ws, bs = [], []
    for epoch in range(epochs):
        current_loss = []
        for i in range(inputs.shape[0]):
            input = tf.convert_to_tensor(inputs[np.newaxis, i].T, dtype=tf.float32)
            target = tf.convert_to_tensor(targets[i], dtype=tf.float32)
            Ws.append(model.W.numpy())
            bs.append(model.b.numpy())
            current_loss.append(loss(model(input), target))

            backward(model, input, target, loss, learning_rate=0.1)

        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
