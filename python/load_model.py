from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "../model/linear.ckpt")
	print("Model restored.")
	print("W : %s" % W.eval())
	print("b : %s" % b.eval())
	x = numpy.asarray([1.0, 4.0, 7.0, 10.0])
	plt.plot(x, sess.run(W) * x +  sess.run(b), label='Fitted line')
	plt.legend()
	plt.show()