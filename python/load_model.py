from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

export_dir='../model/by_graph/linear'

with tf.Session() as sess:  
	tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)
	W = sess.run('weight:0')
	b = sess.run('bias:0')

	print("W: %.2f, b: %.2f" % (W, b))
	# x = numpy.asarray([1.0, 4.0, 7.0, 10.0])
	# print( W * x +  b)
	# plt.plot(x, W * x +  b, label='Fitted line')
	# plt.legend()
	# plt.show()