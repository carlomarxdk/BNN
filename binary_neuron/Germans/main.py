import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution()

import numpy as np

from data_loader import data, labels, inputs, outputs
from model import Model
from train import train

model = Model(n_classes=2, n_features=1)

train(model, data,labels)
