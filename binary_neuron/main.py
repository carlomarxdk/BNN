import tensorflow as tf

tf.enable_eager_execution()
from binary_neuron.data_loader import data, labels
from binary_neuron.model import Model
from binary_neuron.train import train

model = Model()

train(model, data, labels)

# Let's plot it all
# plt.plot(epochs, Ws, 'r',
#          epochs, bs, 'b')
# plt.plot([TRUE_W] * len(epochs), 'r--',
#          [TRUE_b] * len(epochs), 'b--')
# plt.legend(['W', 'b', 'true W', 'true_b'])
# plt.show()
