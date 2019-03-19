import tensorflow as tf

tf.enable_eager_execution()
from data_loader import data, labels
from model import Model
from train import train

model = Model()

train(model, data, labels, epochs=10)
