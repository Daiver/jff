
import tensorflow as tf
import numpy as np
import input_data

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create node for curruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W) # tight weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X # corrupted X

    Y = tf.nn.softplus(tf.matmul(tilde_X, W) + b) # hidden state
    h = tf.nn.dropout(Y, 0.5)

    Z = tf.nn.softplus(tf.matmul(h, W_prime) + b_prime) # reconstructed input
    #Z = tf.nn.softplus(tf.matmul(Y, W_prime) + b_prime) # reconstructed input
    #Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b) # hidden state
    #Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime) # reconstructed input
    return tf.nn.dropout(Z, 0.5)

# build model graph
Z = model(X, mask, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2)) # minimize squared error
#cost = tf.reduce_sum(tf.square(X - Z, 2)) # minimize squared error
train_op = tf.train.AdamOptimizer(0.01).minimize(cost) # construct an optimizer
#train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost) # construct an optimizer

# load MNIST data
mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(20):
    for start, end in zip(range(0, len(trX), 100), range(100, len(trX), 100)):
        input_ = trX[start:end]
        mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
        sess.run(train_op, feed_dict={X: input_, mask: mask_np})

    mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
    print i, sess.run(cost, feed_dict={X: teX, mask: mask_np})

