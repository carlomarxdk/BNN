import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from binary_neuron.utils import enrich_data


def generate_data(num_samples=1000, batch_size=100):
    # [features, labels] = datasets.make_blobs(n_samples=num_samples, shuffle=True, random_state=None)
    # [features, labels] = datasets.make_circles(n_samples=num_samples, shuffle=True, noise=0.01, random_state=None)
    [features, labels] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=0.05, random_state=None)

    features = enrich_data(features)

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)

    min_x, max_x = np.min([features[:, 0]]), np.max([features[:, 0]])
    min_y, max_y = np.min([features[:, 1]]), np.max([features[:, 1]])
    sizes = [min_x, max_x, min_y, max_y]
    return dataset, [features, labels, sizes]


if __name__ == "__main__":
    _, [features, labels, sizes] = generate_data()
    plt.scatter(features[:, 0], features[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
    plt.show()
