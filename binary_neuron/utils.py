import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


@tf.custom_gradient
def hard_tanh(x):
    def grad(dy):
        return dy

    return tf.clip_by_value(x, -1., 1.), grad


@tf.custom_gradient
def binarize(x):
    def grad(dy):
        return dy

    return tf.sign(x), grad


@tf.custom_gradient
def round(x):
    def grad(dy):
        return dy

    return tf.round(x), grad


def enrich_data(features):
    return np.asarray([features[:, 0],
                       features[:, 1],
                       features[:, 0] * features[:, 1],
                       np.sin(features[:, 0]),
                       np.cos(features[:, 1]),
                       np.square(features[:, 0]),
                       np.square(features[:, 1])
                       ]).T


def plotRnnResults(model, dataset, trainX, trainY, testX, testY, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
