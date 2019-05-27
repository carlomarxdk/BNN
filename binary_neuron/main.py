import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from binary_neuron.utils import plotRnnResults

tf.enable_eager_execution()
from binary_neuron.classification.model import ClassificationModel, BaselineClassificationModel
from binary_neuron.classification.data_loader import classification_dataset, classification_dataset_no_batch
from binary_neuron.rnn.data_loader import RNN_dataset
from binary_neuron.rnn.model import RNNModel, BaselineRNNModel
from binary_neuron.rnn.train import train, plot_grid

##############################################################################
########################## Classification ####################################
##############################################################################

# [features, labels, sizes] = classification_dataset_no_batch(num_samples=15000, batch_size=20)
# labels = np.expand_dims(labels, 2)
#
# model = ClassificationModel(classes=2)
# # model = BaselineClassificationModel(classes=2)
# # train(model, dataset, optimizer, epochs=100, print_freq=20, sizes=sizes)
# optimizer = tf.keras.optimizers.Adam(5e-4)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
#
# losses = []
# for i in range(40):
#     res = model.fit(x=features, y=labels, batch_size=50)
#     loss = res.history['loss'][0]
#     if loss < 0.2:
#         break
#     losses.append(loss)
#
# plt.plot(losses)
# plt.show()
#
# # Plotting
#
# dataset, [features, labels, _] = classification_dataset(num_samples=150, batch_size=20)
#
# colors = {0: '#de0000', 1: '#1f2b7b', 2: '#cfde3c'}
# _labels = [colors[label] for label in labels.numpy()]
# plot_grid(model, sizes=sizes)
#
# preds = np.argmax(model(features).numpy(), axis=1)
# _preds = [colors[pred] for pred in preds]
# plt.scatter(features[:, 0], features[:, 1], s=40, c=_preds, cmap=plt.cm.Spectral)
# plt.show()
#
# accuracy = round(((1. - (np.sum(np.not_equal(preds, labels.numpy())) / len(preds))) * 100), 2)
# print(f'Accuracy: {accuracy}%')

##############################################################################
##############################################################################
##############################################################################

##############################################################################
################################# LSTM #######################################
##############################################################################

LOOK_BACK = 3

[trainX, trainY, testX, testY, dataset] = RNN_dataset(look_back=LOOK_BACK)
# model = RNNModel(units=10, look_back=LOOK_BACK)
model = BaselineRNNModel(look_back=LOOK_BACK)

model.compile(loss='mean_squared_error', optimizer='adam')

losses = []
for i in range(200):
    res = model.fit(trainX, trainY, batch_size=15, verbose=2)
    loss = res.history['loss'][0]
    if loss < 0.0035:
        print('Stopping early')
        break
    losses.append(loss)

plt.plot(losses)
plt.show()

plotRnnResults(model=model, trainX=trainX, trainY=trainY, testX=testX, testY=testY, dataset=dataset,
               look_back=LOOK_BACK)

##############################################################################
##############################################################################
##############################################################################
