import tensorflow as tf


class Model(object):
    def __init__(self):
        self.W = tf.Variable(tf.random.uniform([10, 2], dtype=tf.float32))
        self.b = tf.Variable(tf.random.uniform([10], dtype=tf.float32))

    def params(self):
        return [self.W, self.b]

    def update(self, gradients, learning_rate):
        # for idx, param in self.params():
        #     print(param.numpy())
        self.W.assign_sub(gradients[0] * learning_rate)
        self.b.assign_sub(gradients[1] * learning_rate)

    def __call__(self, x):
        return tf.linalg.matmul(self.W, x) + self.b


if __name__ == "__main__":
    tf.enable_eager_execution()

    # graph = tf.Graph()
    # sess = tf.Session()  # graph=graph)
    # with sess.as_default():
    model = Model()
    print(model.W.numpy())
    model.update([tf.random.uniform([10]), tf.random.uniform([10])], 1)
    print(model.W.numpy())
