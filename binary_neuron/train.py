import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_grid(model, resolution=10):
    grid = np.asarray([(i / 100, j / 100) for j in range(-50, 100, resolution) for i in range(-100, 200, resolution)])
    colors = {0: '#de0000', 1: '#1f2b7b'}
    preds = np.argmax(model(tf.convert_to_tensor(grid)).numpy(), axis=1)
    labels = [colors[pred] for pred in preds]

    plt.scatter(grid[:, 0], grid[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
    plt.show()


def loss(X, Y):
    return tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=X)


def backward(model, X, Y, optimizer):
    with tf.GradientTape() as t:
        pred = model(X)
        current_loss = loss(pred, Y)
        grads = t.gradient(current_loss, model.params())
        optimizer.apply_gradients(zip(grads, model.params()))
    return current_loss


def train(model, data, optimizer, epochs=10, batch_size=50, print_freq=5):
    losses = np.zeros(epochs)
    for epoch in range(epochs):
        current_loss = []
        for batch, (X, Y) in enumerate(data):
            if batch <= batch_size:
                current_loss.append(backward(model, X, Y, optimizer))
            else:
                break

        print(f'Epoch {str(epoch).zfill(2)}: loss={round(np.asarray(current_loss).mean(), 3)}')
        losses[epoch] = np.asarray(current_loss).mean()
        if epoch % print_freq == 0:
            plot_grid(model, resolution=5)

    plt.plot(losses)
    plt.show()
