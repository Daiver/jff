import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def LReLU(x, alpha=0.001):
    return tf.maximum(alpha*x,x)

def model(
        X, 
        w_h, b_h, 
        w_h2, b_h2, 
        w_o, b_o, 
        p_drop_input, 
        p_drop_hidden): # 
    X = tf.nn.dropout(X, p_drop_input)
    h = LReLU(tf.matmul(X, w_h) + b_h)
    #h = tf.nn.softplus(tf.matmul(X, w_h) + b_h)
    #h = tf.nn.relu(tf.matmul(X, w_h))
    h = tf.nn.dropout(h, p_drop_hidden)

    h2 = LReLU(tf.matmul(h, w_h2) + b_h2)
    #h2 = tf.nn.softplus(tf.matmul(h, w_h2) + b_h2)
    #h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_drop_hidden) + h

    return tf.matmul(h2, w_o) + b_o


mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 1000])
b_h = init_weights([1000])
w_h2 = init_weights([1000, 1000])
b_h2 = init_weights([1000])
w_o = init_weights([1000, 10])
b_o = init_weights([10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(
        X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
#train_op = tf.train.AdamOptimizer(  0.0002, 0.8).minimize(cost)
train_op = tf.train.RMSPropOptimizer(0.0002, 0.5).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    batchSize = 64
    for start, end in zip(range(0, len(trX), batchSize), range(batchSize, len(trX), batchSize)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_input: 0.5, p_keep_hidden: 0.3})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                     p_keep_input: 1.0,
                                                     p_keep_hidden: 1.0}))
