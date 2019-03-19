import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from binary_neuron.Germans.utils import mini_batch

def loss(output, target):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    return tf.reduce_mean(tf.square(output - target))

def backward(model, inputs, outputs, loss, learning_rate):
    with tf.GradientTape() as t:
        loss_value = loss(model(inputs), outputs)

    gradients = t.gradient(loss_value, model.params())
    model.update(gradients, learning_rate)

def train(model, inputs, targets):

    losses = np.zeros(model.epochs)
    for epoch in range(model.epochs):
        current_loss = []
        batch_inputs, batch_outputs = mini_batch(inputs, targets)
        for indx_, batch in enumerate(batch_inputs):
            for idx, input in enumerate(batch):
                target = batch_outputs[indx_][idx]
                current_loss.append(loss(model(input), target))

                backward(model, input, target, loss, learning_rate=model.learning_rate)

        print('Epoch %2d: loss=%2.5f' %
              (epoch, np.asarray(current_loss).mean()))
        losses[epoch] = np.asarray(current_loss).mean()

    targets = np.asarray([model(input).numpy() for input in inputs]).flatten()
    plt.scatter(inputs[:, 0], inputs[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()