import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

from binary_neuron.Germans.data_loader import data, labels, inputs, outputs
from binary_neuron.Germans.model import Model
from binary_neuron.Germans.train import train
from binary_neuron.Germans.utils import mini_batch

model = Model(n_classes=2, n_features=2)

train(model, data,labels)


#x, y = mini_batch(inputs, outputs)
#for batch in x:
#    for indx, value in enumerate(batch):
#        print(value)