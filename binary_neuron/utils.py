import tensorflow as tf
from tensorflow.python.framework import ops


def binarize(tensor, dtype=tf.float32):
    ones = tf.ones(tensor.shape, dtype)
    twos = tf.cast(tf.fill(tensor.shape, 2), dtype)

    tensor = tf.greater_equal(tensor, 0)  # True, False
    tensor = tf.cast(tensor, dtype)  # 0, 1
    tensor = tf.multiply(tensor, twos)  # 0, 2
    return tf.subtract(tensor, ones)  # -1, 1


def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

        # For Tensorflow v0.11 and below use:
        # with g.gradient_override_map({"Floor": "Identity"}):
        #    return tf.round(x, name=name)


def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (identity).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


def passThroughSigmoid(x, slope=1):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)


def binaryStochastic_ST(x, slope_tensor=None, pass_through=True, stochastic=True):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
    step function (if stochastic == False). Uses the straight through estimator.
    See https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function
        for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1 or 0;
        if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
        Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if pass_through:
        p = passThroughSigmoid(x)
    else:
        p = tf.sigmoid(slope_tensor * x)

    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p)


def hard_tanh(x):
    return tf.clip_by_value(
        x,
        tf.constant(-1, dtype=x.dtype),
        tf.constant(1, dtype=x.dtype),
        name="Hard tanh"
    )
