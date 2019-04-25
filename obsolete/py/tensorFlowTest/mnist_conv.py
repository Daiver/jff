from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():
  data_dir ='/tmp/tensorflow/mnist/input_data' 
  sum_dir = 'sumdir'
  mnist = input_data.read_data_sets(data_dir, one_hot=True)


  x = tf.placeholder(tf.float32, [None, 784])
  x_conv = tf.reshape(x, [-1, 28, 28, 1])

  keep_prob = tf.placeholder(tf.float32)

  W1_conv = weight_variable([3, 3, 1, 32])
  b1_conv = bias_variable([32])

  c1 = conv2d(x_conv, W1_conv) + b1_conv
  r1 = tf.nn.relu(c1)
  p1 = max_pool_2x2(r1)
  #f1 = tf.reshape(p1, [-1, 14*14*16])

  W2_conv = weight_variable([5, 5, 32, 64])
  b2_conv = bias_variable([64])

  c2 = conv2d(p1, W2_conv) + b2_conv
  r2 = tf.nn.relu(c2)
  p2 = max_pool_2x2(r2)
  f2 = tf.reshape(p2, [-1, 7*7*64])
  
  W3 = weight_variable([7*7*64, 10])
  b3 = bias_variable([10])
  f3 = tf.matmul(f2, W3) + b3
  y  = f3

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
  #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.InteractiveSession()

  #print ('shape', sess.run(tf.shape(f2), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
  #return

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()

  train_writer = tf.train.SummaryWriter(sum_dir + '/train', sess.graph)
  val_writer   = tf.train.SummaryWriter(sum_dir + '/validate', sess.graph)

  tf.global_variables_initializer().run()

  batch_size = 128
  #batch_size = 16
  best_acc = 0
  for epochInd in range(40):
    for _ in range(mnist.train.num_examples // batch_size // 2):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Test trained model
    batch_xs, batch_ys = mnist.train.next_batch(5000)
    summary, train_acc = sess.run([merged, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    train_writer.add_summary(summary, epochInd)
    train_writer.flush()

    print (epochInd, 'Train acc', train_acc)
    summary, val_acc = sess.run([merged, accuracy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
    val_writer.add_summary(summary, epochInd)
    val_writer.flush()
    if val_acc > best_acc:
      best_acc = val_acc
    print (epochInd, 'Valid acc', val_acc, 'best', best_acc)

  test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  print (epochInd, 'Test acc', test_acc)

  train_writer.close()
  val_writer.close()

main()


