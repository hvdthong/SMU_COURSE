from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def sample_CNN(iter):
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # print mnist.train.images.shape, mnist.test.images.shape
    sess = tf.InteractiveSession()

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(iter):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # cross_entropy_value = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # print("step %d, training accuracy %g, cross entropy %f" % (i, train_accuracy, cross_entropy_value))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def modify_CNN(iter):
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # tf.matmul: using to multiple two matrix

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(iter):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # cross_entropy_value = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # print("step %d, training accuracy %g, cross entropy %f" % (i, train_accuracy, cross_entropy_value))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    # sample_CNN(15000)
    modify_CNN(1000)
    modify_CNN(5000)
    modify_CNN(10000)
    modify_CNN(15000)

    ###################################################################
    # For debugging
    # mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # # print mnist.train.images.shape, mnist.test.images.shape
    # sess = tf.InteractiveSession()
    # x_image = tf.reshape(mnist.train.images, [-1, 28, 28, 1])
    #
    # W_conv1 = weight_variable([5, 5, 1, 16])
    # b_conv1 = bias_variable([16])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # # h_pool1 = max_pool_2x2(h_conv1)
    #
    # W_conv2 = weight_variable([5, 5, 16, 32])
    # b_conv2 = bias_variable([32])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    #
    # W_conv3 = weight_variable([5, 5, 32, 64])
    # b_conv3 = bias_variable([64])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)
    #
    # W_fc1 = weight_variable([14 * 14 * 64, 2056])
    # b_fc1 = bias_variable([2056])
    # h_pool3_flat = tf.reshape(h_pool3, [-1, 14 * 14 * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # tf.matmul: using to multiple two matrix
    #
    # sess.run(tf.initialize_all_variables())
    # print "image size: ", x_image.eval().shape
    # print "first convolution layer: ", h_conv1.eval().shape
    # print "second convolution layer: ", h_conv2.eval().shape
    # print "third convolution layer: ", h_conv3.eval().shape
    # print "max_pooling third layer:", h_pool.eval().shape
    # print "flating: ", h_pool_flat.eval().shape
    # print "fully connected layer: ", h_fc1.eval().shape
    # print "output of final layer: ", y_conv.eval().shape
    ###################################################################

    ###################################################################
    # For adjusting the CNN
    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    # x = tf.placeholder(tf.float32, [None, 784])
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    #
    # x_image = tf.reshape(mnist.train.images, [-1, 28, 28, 1])
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #
    # with tf.device('/cpu:0'):
    #     W_conv2 = weight_variable([5, 5, 32, 64])
    #     b_conv2 = bias_variable([64])
    #     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    #
    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = weight_variable([128])
    # h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    # h_pool = max_pool_2x2(h_conv3)  # image: 14x14, depth: 128
    #
    # W_fc1 = weight_variable([14 * 14 * 128, 1024])
    # b_fc1 = bias_variable([1024])
    # h_pool_flat = tf.reshape(h_pool, [-1, 14 * 14 * 128])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)  # tf.matmul: using to multiple two matrix
    #
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #
    # sess.run(tf.initialize_all_variables())
    # print "image size: ", x_image.eval().shape
    # print "first convolution layer: ", h_conv1.eval().shape
    # print "second convolution layer: ", h_conv2.eval().shape
    # print "third convolution layer: ", h_conv3.eval().shape
    # print "max_pooling third layer:", h_pool.eval().shape
    # print "flating: ", h_pool_flat.eval().shape
    # print "fully connected layer: ", h_fc1.eval().shape
    # print "output of final layer: ", y_conv.eval().shape
    ###################################################################