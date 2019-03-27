from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf

num_samples = 1500
noise = 0.1

##[X, y] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=noise, random_state=None)

[X, y] = datasets.make_moons(num_samples, shuffle=True, noise=noise, random_state=None)

X_train = X[:1000].astype('float32')
X_validation = X[1000:].astype('float32')

y_train = y[:1000].astype('int32')
y_validation = y[1000:].astype('int32')

num_features = X_train.shape[-1]
num_output = 2