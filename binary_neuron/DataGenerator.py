from sklearn import datasets

class DataGenerator():
    def __init__(self, num_samples, noise):
        [self.X, self.y] = datasets.make_moons(num_samples, shuffle=True, noise=noise, random_state=None)
        #X_validation = X[1000:].astype('float32')
        #y_validation = y[1000:].astype('int32')

    def train(self, num_observations):
        return [self.X[:num_observations].astype('float32'), self.y[:num_observations].astype('int32')]
    def num_features(self):
        return  self.X.shape[-1]
