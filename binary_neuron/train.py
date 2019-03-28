import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot(inputs, model):
    targets = np.asarray([model(input).numpy() for input in inputs]).flatten()
    plt.scatter(inputs[:, 0], inputs[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()


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

    _gradients = []
    for gradient in gradients:
        max = tf.norm(gradient)
        if not tf.equal(max, 0):
            _gradients.append(gradient / max)
        else:
            _gradients.append(gradient)
    gradients = _gradients

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

            backward(model, input, target, loss, learning_rate=1e-3)

        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
        losses[epoch] = np.asarray(current_loss).mean()
        if epoch % 5 == 0:
            grid = np.asarray([(i / 100, j / 100) for j in range(-50, 100, 10) for i in range(-100, 200, 10)])
            plot(grid, model)

    plt.plot(losses)
    plt.show()

    plot(inputs, model)

    grid = np.asarray([(i / 100, j / 100) for j in range(-50, 100, 5) for i in range(-100, 200, 5)])
    plot(grid, model)
