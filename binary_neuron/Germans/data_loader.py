from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples = 1000
noise = 0.1

[data, labels] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=noise, random_state=None)

# Tensors representing the data and labels
inputs = tf.constant(data, dtype=tf.float32)
outputs = tf.constant(labels, dtype=tf.float32)
