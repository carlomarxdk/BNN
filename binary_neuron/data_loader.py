from sklearn import datasets
import matplotlib.pyplot as plt

num_samples = 100
[data, labels] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=False, random_state=None)
# data = tf.convert_to_tensor(data_raw.T, dtype=tf.float32)
# labels = tf.convert_to_tensor(labels_raw, dtype=tf.float32)

if __name__ == "__main__":
    plt.scatter(data[:, 0], data[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
    plt.show()
