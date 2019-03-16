import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loss(output, target):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    return tf.reduce_mean(tf.square(output - target))


def backward(model, inputs, targets, loss, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), targets)
    # print('Inputs: ', inputs.numpy())
    # print('Target: ', targets.numpy())
    # print('Predictions: ', model(inputs).numpy())
    # print('Losses: ', current_loss)
    # print('Params: ', model.params())

    gradients = t.gradient(current_loss, model.params())

    # print('Gradients: ', gradients)
    model.update(gradients, learning_rate)


def train(model, inputs, targets, epochs=10):
    plt.scatter(inputs[:, 0], inputs[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()
    losses = np.zeros(epochs)
    for epoch in range(epochs):
        current_loss = []
        for idx, input in enumerate(inputs):
            target = targets[idx]
            current_loss.append(loss(model(input), target))

            backward(model, input, target, loss, learning_rate=1e-4)

        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
        losses[epoch] = np.asarray(current_loss).mean()
    plt.plot(losses)
    plt.show()

    targets = np.asarray([model(input).numpy() for input in inputs]).flatten()
    plt.scatter(inputs[:, 0], inputs[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()
