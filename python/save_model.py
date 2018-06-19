from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

learning_rate = 0.01
training_epochs = 1000
display_step = 50
train_X = numpy.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train_Y = numpy.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_samples = train_X.shape[0]
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})

    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

    save_path = saver.save(sess, "../model/linear.ckpt")
    print("Model saved in path: %s" % save_path)
    print("W : %s" % sess.run(W))
    print("b : %s" % sess.run(b))
