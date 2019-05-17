import pandas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from binary_neuron.utils import enrich_data
from pandas import read_csv


def classification_dataset(num_samples=1000, batch_size=100):
    # [features, labels] = datasets.make_blobs(n_samples=num_samples, shuffle=True, random_state=None)
    # [features, labels] = datasets.make_circles(n_samples=num_samples, shuffle=True, noise=0.01, random_state=None)
    [features, labels] = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=0.05, random_state=None)

    features = enrich_data(features)

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)

    min_x, max_x = np.min([features[:, 0]]), np.max([features[:, 0]])
    min_y, max_y = np.min([features[:, 1]]), np.max([features[:, 1]])
    sizes = [min_x, max_x, min_y, max_y]
    return dataset, [features, labels, sizes]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def RNN_dataset(look_back=1):
    dataframe = read_csv('airline_passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return [trainX, trainY]


if __name__ == "__main__":
    RNN_dataset()
