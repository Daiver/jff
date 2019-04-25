from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  data_dir ='/tmp/tensorflow/mnist/input_data' 
  mnist = input_data.read_data_sets(data_dir, one_hot=True)

  hiddenLayerSize = 500

  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.truncated_normal([784, hiddenLayerSize], stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([hiddenLayerSize], stddev=0.1))
  f1 = tf.nn.relu(tf.matmul(x, W1) + b1)
  W2 = tf.Variable(tf.truncated_normal([hiddenLayerSize, 10], stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
  f2 = tf.nn.relu(tf.matmul(f1, W2) + b2)
  y  = f2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  #train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.global_variables_initializer().run()
  batch_size = 32
  #batch_size = 16
  bestAcc = 0
  for epochInd in range(20):
    for _ in range(mnist.train.num_examples // batch_size):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print (epochInd, 'Train acc', sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
    testAcc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    if testAcc > bestAcc:
      bestAcc = testAcc
    print (epochInd, 'Test acc', testAcc, 'best', bestAcc)



main(None)


