import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from binary_neuron.utils import binarize, hard_tanh
from tensorflow.python.keras import backend as K


class BinaryLinearLayer(tf.keras.layers.Layer):
    def __init__(self, num_out, binarize_input=True):
        super(BinaryLinearLayer, self).__init__()
        self.num_out = num_out
        self.binarize_input = binarize_input

    def build(self, input_shape):
        self.weight = tf.get_variable('weight', [int(input_shape[-1]), self.num_out],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        super(BinaryLinearLayer, self).build(input_shape)

    def call(self, x):
        if self.binarize_input:
            x = binarize(x)
        weight = binarize(self.weight)
        return tf.matmul(x, weight)


class HardTanH(tf.keras.layers.Layer):
    def call(self, x):
        return hard_tanh(x)


class SoftMax(tf.keras.layers.Layer):
    def call(self, x):
        return tf.nn.softmax(x)


class BinaryLSTMCell(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 **kwargs):
        super(BinaryLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = hard_tanh  # activations.get(activation)
        self.recurrent_activation = hard_tanh  # activations.get(recurrent_activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.state_size = [self.units, self.units]
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)

        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        (x_i, x_f, x_c, x_o) = x

        W_i = binarize(self.recurrent_kernel[:, :self.units])
        W_f = binarize(self.recurrent_kernel[:, self.units:self.units * 2])
        W_c = binarize(self.recurrent_kernel[:, self.units * 2:self.units * 3])
        W_o = binarize(self.recurrent_kernel[:, self.units * 3:])

        # f = sigmoid(W_f*[h_tm1, x])
        f = self.recurrent_activation(x_i + K.dot(h_tm1, W_f))

        # i = sigmoid(W_i*[h_tm1, x])
        i = self.recurrent_activation(x_f + K.dot(h_tm1, W_i))

        # c = f * c_tm1 + i * (tanh(W_c[h_tm1, x]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, W_c))

        # sigmoid(W_o * [h_tm1, x])
        o = self.recurrent_activation(x_o + K.dot(h_tm1, W_o))

        return c, o

    def call(self, inputs, states):
        h_tm1 = hard_tanh(states[0])  # previous memory state
        c_tm1 = hard_tanh(states[1])  # previous carry state

        W_k_i = binarize(self.kernel[:, :self.units])
        W_k_f = binarize(self.kernel[:, self.units:self.units * 2])
        W_k_c = binarize(self.kernel[:, self.units * 2:self.units * 3])
        W_k_o = binarize(self.kernel[:, self.units * 3:])

        x_i = hard_tanh(K.dot(inputs, W_k_i))
        x_f = hard_tanh(K.dot(inputs, W_k_f))
        x_c = hard_tanh(K.dot(inputs, W_k_c))
        x_o = hard_tanh(K.dot(inputs, W_k_o))

        x = (x_i, x_f, x_c, x_o)
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

        h = hard_tanh(o * self.activation(c))
        return h, [h, c]


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return array_ops.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)
