from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main():
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()
    for i in range(15000):
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs, batch_ys = mnist.train.next_batch(64)
        if i % 64 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            cross_entropy_value = cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print("step %d, training accuracy %g, cross entropy %f" % (i, train_accuracy, cross_entropy_value))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/data',
    #                     help='Directory for storing data')
    # FLAGS = parser.parse_args()
    # tf.app.run()
    main()