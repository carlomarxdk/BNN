from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf


def generate_data(num_samples=1000, batch_size=100):
    [features, labels] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=0.01, random_state=None)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    return dataset, [features, labels]


if __name__ == "__main__":
    _, [features, labels] = generate_data()
    plt.scatter(features[:, 0], features[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
    plt.show()
