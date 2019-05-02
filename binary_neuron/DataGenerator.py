from sklearn import datasets
import numpy as np

class DataGenerator():
    def __init__(self, num_samples, noise):
        [self.X, self.y] = datasets.make_moons(num_samples, shuffle=True, noise=noise, random_state=None)
        #X_validation = X[1000:].astype('float32')
        #y_validation = y[1000:].astype('int32')


        self.X = np.transpose(np.asarray([self.X[:, 0],
                         self.X[:, 1],
                         self.X[:, 0] * self.X[:, 1],
                         np.sin(self.X[:, 0]),
                         np.cos(self.X[:, 1]),
                         np.square(self.X[:, 0]),
                         np.square(self.X[:, 1])
                         ]))
    def train(self, num_observations):
        return [self.X[:num_observations].astype('float32'), self.y[:num_observations].astype('int32')]
    def num_features(self):
        return  self.X.shape[-1]
