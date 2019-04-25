import input_data
import tensorflow as tf

if __name__ == '__main__':
    mnist = input_data.read_data_sets("/home/daiver/Downloads/MNIST_data/", one_hot=True)
    x = tf.placeholder("float", [None, 784])

    hiddenLayerSize = 1000
    hiddenLayer2Size = 500
    hiddenLayer3Size = 250
    hiddenLayer4Size = 100

    W1 = tf.Variable(tf.random_normal([784,hiddenLayerSize]))*0.01
    W2 = tf.Variable(tf.random_normal([hiddenLayerSize,hiddenLayer2Size]))*0.01
    W3 = tf.Variable(tf.random_normal([hiddenLayer2Size,hiddenLayer3Size]))*0.01
    W4 = tf.Variable(tf.random_normal([hiddenLayer3Size,hiddenLayer4Size]))*0.01
    W5 = tf.Variable(tf.random_normal([hiddenLayer4Size,10]))*0.01

    b1 = tf.Variable(tf.random_normal([hiddenLayerSize]))*0.01
    b2 = tf.Variable(tf.random_normal([hiddenLayer2Size]))*0.01
    b3 = tf.Variable(tf.random_normal([hiddenLayer3Size]))*0.01
    b4 = tf.Variable(tf.random_normal([hiddenLayer4Size]))*0.01
    b5 = tf.Variable(tf.random_normal([10]))*0.01

    #y1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    y1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    y2 = tf.nn.softplus(tf.matmul(y1,W2) + b2)
    y3 = tf.nn.softplus(tf.matmul(y2,W3) + b3)
    y4 = tf.nn.softplus(tf.matmul(y3,W4) + b4)
    y5 = tf.nn.softmax(tf.matmul(y4,W5) + b5)

    resAct = y5

    y_ = tf.placeholder("float", [None,10])

    regConstant = 0.1
    regFunc = regConstant * (
            tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) +
            tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) 
            + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2) + 
            tf.nn.l2_loss(b3) + tf.nn.l2_loss(b4) + tf.nn.l2_loss(b5))

    errorFunctional = tf.reduce_sum(tf.square(y_ - resAct)) + regFunc

    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(errorFunctional)
    train_step = tf.train.AdamOptimizer(0.05).minimize(errorFunctional)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(resAct,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    for i in range(50000):
        batch_xs, batch_ys = mnist.train.next_batch(211)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 10 == 0:
            print i, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 500 == 0:
            print 'Cur test'
            print sess.run(
                    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

