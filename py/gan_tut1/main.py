import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def weight_variable(shape, name=None):
    #initial = tf.truncated_normal(shape, stddev=np.linalg.norm(shape))
    n_in = 1
    for x in shape[:-1]:
        n_in *= x
    n_out = shape[-1]
    #initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0)/np.sqrt(n_in + n_out))
    initial = tf.random_normal(shape, stddev=1.0)
    #initial = tf.truncated_normal(shape, stddev=1.0/np.sqrt(n_in))
    #initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    #initial = tf.truncated_normal(shape, stddev=0.0)
    return tf.Variable(initial, name=name)

def mkGenerator(input, weights):
    l1 = tf.nn.softplus(tf.matmul(input, weights['w1'] + weights['b1']))
    #l1 = tf.nn.elu(tf.matmul(input, weights['w1'] + weights['b1']))
    #l1 = tf.nn.tanh(tf.matmul(input, weights['w1'] + weights['b1']))
    return tf.matmul(l1, weights['w2']) + weights['b2']

def mkDiscriminator(input, weights):
    l1 = tf.nn.tanh(tf.matmul(input, weights['w1']) + weights['b1'])
    l2 = tf.nn.tanh(tf.matmul(l1,weights['w2']) + weights['b2'])
    l3 = tf.sigmoid(tf.matmul(l2,weights['w3']) + weights['b3'])
    return l3
    l4 = tf.sigmoid(tf.matmul(l3,weights['w4']) + weights['b4'])
    return l4


if __name__ == '__main__':
    center = 2.0
    std = 1.0
    n_points = 100000
    data = np.random.normal(center, std, [n_points, 1])
    plt.hist(data, bins=50)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.clf()
    #data2 = np.random.normal(2, 1, [n_points])
    #plt.hist(data2, bins=50)
    
    #plt.hist(data, bins=50)
    #plt.grid(True)
    #plt.show()

    num_hidden = 4

    w_gen1 = weight_variable([1, num_hidden])  
    b_gen1 = bias_variable([num_hidden])  
    w_gen2 = weight_variable([num_hidden, 1])  
    b_gen2 = bias_variable([1])  

    gen_vars = {
        'w1': w_gen1, 'b1': b_gen1,
        'w2': w_gen2, 'b2': b_gen2
        }

    num_hidden = 8
    w_dis1 = weight_variable([1, num_hidden])  
    b_dis1 = bias_variable([num_hidden])  
    w_dis2 = weight_variable([num_hidden, num_hidden])  
    b_dis2 = bias_variable([num_hidden])  
    w_dis3 = weight_variable([num_hidden, 1])  
    b_dis3 = bias_variable([1])  
    dis_vars = {
            'w1': w_dis1,
            'b1': b_dis1,
            'w2': w_dis2,
            'b2': b_dis2,
            'w3': w_dis3,
            'b3': b_dis3
        }

    z = tf.placeholder(tf.float32, shape=(None, 1))

    G = mkGenerator(z, gen_vars)

    x  = tf.placeholder(tf.float32, shape=(None, 1))
    D1 = mkDiscriminator(x, dis_vars)
    D2 = mkDiscriminator(G, dis_vars)

    sess = tf.InteractiveSession()
    
    loss_d  = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
    loss_g  = tf.reduce_mean(-tf.log(D2))

    learning_rate = 0.003
    optstep_d = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_d, var_list=dis_vars.values())
    optstep_g = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_g, var_list=gen_vars.values())
    #optstep_d = tf.train.AdamOptimizer(learning_rate).minimize(loss_d, var_list=dis_vars.values())
    #optstep_g = tf.train.AdamOptimizer(learning_rate).minimize(loss_g, var_list=gen_vars.values())
    tf.global_variables_initializer().run()
    batch_size = 64
    for iterInd in xrange(20000):
        for _ in xrange(5):
            batch_z = np.random.normal(size=[batch_size, 1]) * 0.01
            batch_x = data[np.random.choice(data.shape[0], batch_size, replace=False), :]
            loss_d_val = sess.run([loss_d, optstep_d], feed_dict={x: batch_x, z: batch_z})[0]
        for _ in xrange(1):
            batch_z = np.random.normal(size=[batch_size, 1]) * 0.01
            loss_g_val = sess.run([loss_g, optstep_g], feed_dict={z: batch_z})[0]

        if iterInd % 10 == 0:
            print iterInd, '> d', loss_d_val, 'g', loss_g_val, 'rate', learning_rate

        if iterInd % 150 == 0:
            learning_rate *= 0.95
            del optstep_d
            del optstep_g
            optstep_d = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                                            loss_d, var_list=dis_vars.values())
            optstep_g = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                                            loss_g, var_list=gen_vars.values())

        if iterInd % 5 == 0 and iterInd > 0:
            curData = sess.run(G, feed_dict = {
                z: np.random.normal(size=[n_points, 1])
                })


            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.hist(data, bins=50)
            plt.hist(curData, bins=50)
            plt.grid(True)
            #plt.show()
            plt.savefig('%d_.png' % iterInd)
            plt.clf()

    curData = sess.run(G, feed_dict = {
        z: np.random.normal(size=[n_points, 1])
        })

    plt.hist(data, bins=50)
    plt.hist(curData, bins=50)
    plt.grid(True)
    plt.show()

