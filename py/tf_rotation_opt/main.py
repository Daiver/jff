import numpy as np
import tensorflow as tf

#def rotation(quaternion):
    #tmp = 

if __name__ == '__main__':
    points_ = tf.placeholder("float", [None, 3])
    translation_ = tf.Variable(tf.zeros([3]))
    #translation_ = tf.placeholder("float", [3])
    newPoints = points_ + translation_
    targets_  = tf.placeholder("float", [None, 3])
    indices = [2]
    diff = tf.gather(newPoints, indices) - targets_
    err = tf.reduce_sum(diff * diff)

    points = np.array([
        [1, 2, 3],
        [5, 5, 5],
        [0, 0, 5]
        ], dtype=np.float32)

    targets = np.array([
        #[2, 4, 6],
        [1, 2, 9]
        ], dtype=np.float32)


    translation = np.array([
        1, 2, 3
        ], dtype=np.float32)

    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    feed_dict={
        points_: points, 
        #translation_: translation,
        targets_: targets
    }

    #print session.run(tf.reshape(diff, [-1]), feed_dict = feed_dict)
    print session.run(tf.gradients(tf.reshape(diff, [-1]), translation_), feed_dict = feed_dict)

    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(err)
    #for i in xrange(1, 100):
        #print session.run(err, feed_dict)
        #session.run(train_step, feed_dict = feed_dict)

    #print session.run(err, feed_dict)
    #print session.run(translation_)
