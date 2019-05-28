import tensorflow as tf
from tensorflow import contrib
from utils import *
from Logs import *
import numpy as np
import matplotlib.pyplot as plt



summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis=10)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()

def train(model, inputs, targets, in_, tar_):
    min = np.min(inputs)
    max = np.max(inputs)
    dataset = data(inputs, targets, model.batch_size)
    validation = data(in_, tar_, model.batch_size)
    tfe = contrib.eager

    optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate )

    global_step = tf.Variable(0)

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    val_accuracy_results = []

    for epoch in range(model.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        epoch_val_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in dataset:
            y = tf.cast(tf.squeeze(y), dtype=tf.int32)
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            #if epoch > 10: print(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(model.predictions(x), y)


        for x,y in validation:
            y = tf.cast(tf.squeeze(y), dtype=tf.int32)
            epoch_val_accuracy(model.predictions(x), y)




        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        val_accuracy_results.append(epoch_val_accuracy.result())

        global_step.assign_add(1)
        #log_prediction(model, range=(min,max))


        print("Epoch {:03d}: Loss: {:.3f} | Accuracy: {:.3f} | Validation: {:.3f}".format(epoch,epoch_loss_avg.result(),
                                                  epoch_accuracy.result(), epoch_val_accuracy.result()))