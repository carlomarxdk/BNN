import tensorflow as tf
from tensorflow import contrib
from utils import *
from Logs import *
import numpy as np

summary_writer = tf.contrib.summary.create_file_writer('logs', flush_millis=10000)
summary_writer.set_as_default()
global_step = tf.train.get_or_create_global_step()

def train(model, inputs, targets):
    dataset = data(inputs, targets, model.batch_size)
    tfe = contrib.eager

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01 )

    global_step = tf.Variable(0)

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(model.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model.forward(x, in_training_mode=False), axis=1, output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        global_step.assign_add(1)
        log_prediction(model)

        print("Epoch {:03d}: Loss: {:.3f} | Accuracy: {:.3f}".format(epoch,epoch_loss_avg.result(),
                                                  epoch_accuracy.result()))