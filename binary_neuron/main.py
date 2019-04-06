import tensorflow as tf
tf.enable_eager_execution()

import sklearn
import numpy as np
from Train import *
from Model import *

from DataGenerator import *

data = DataGenerator(num_samples=1500, noise=0.2)
[X_train, y_train] = data.train(num_observations=1000)
num_feature = data.num_features()

model = Model(n_classes=2, n_features=2, learning_rate=1e-3, epochs=100)
train(model, X_train, y_train)
