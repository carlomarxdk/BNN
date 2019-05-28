from sklearn import datasets
import numpy as np
import tensorflow.keras.datasets as data

class DataGenerator():
    def __init__(self):
        [self.train, self.test] = data.cifar10.load_data()
    def train(self, num_observations=1000):
        return self.train# [Image, label]