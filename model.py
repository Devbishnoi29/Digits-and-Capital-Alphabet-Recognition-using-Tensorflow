# Programmer : Dev Bishnoi

# This module is constructing a deep neural network with convolutional and fully connected layers.

import tensorflow as tf

width = 28
height = 28
N = 36 # No of classes
flatten_size = width * height
rate = 4e-4

def model(intput_x, input_y):

	with tf.name_scope("reshapeTo4d"):
		reshaped = tf.reshape(intput_x, [-1, width, height, 1], name = 'reshaped')

	with tf.name_scope("paramsAtL1"):
		lyr1_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 64], dtype = tf.float32), name = "lyr1_w")
		lyr1_b = tf.Variable(tf.constant(0.1, shape=[64], dtype = tf.float32), name = "lyr1_b")

	with tf.name_scope("Layer1"):
		lyr1_opt = tf.nn.relu(tf.nn.conv2d(reshaped, lyr1_w, strides = [1, 1, 1, 1], padding = 'SAME') + lyr1_b, name = "reluLyr1")

	with tf.name_scope("maxpoolingAtLayer1"):
		lyr1_mxopt = tf.nn.max_pool(lyr1_opt, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'mxpoolLyr1')

	with tf.name_scope("paramsAtL2"):
		lyr2_w = tf.Variable(tf.truncated_normal(shape = [5, 5, 64, 128], dtype = tf.float32), name = "lyr2_w")
		lyr2_b = tf.Variable(tf.constant(0.1, shape=[128], dtype = tf.float32), name = "lyr2_b")

	with tf.name_scope("Layer2"):
		lyr2_opt = tf.nn.relu(tf.nn.conv2d(lyr1_mxopt, lyr2_w, strides = [1, 1, 1, 1], padding = 'SAME') + lyr2_b, name = "reluLyr2")

	with tf.name_scope("maxpoolingAtLayer2"):
		lyr2_mxopt = tf.nn.max_pool(lyr2_opt, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = "mxpoolLyr2")

	with tf.name_scope("FlattenForFCL"):
		flatten_opt = tf.reshape(lyr2_mxopt, [-1, 7 * 7 * 128], name = "Flatten")

	with tf.name_scope("paramsAtL3"):
		lyr3_w = tf.Variable(tf.truncated_normal(shape = [7 * 7 * 128, 864], dtype = tf.float32), name = "lyr3_w")
		lyr3_b = tf.Variable(tf.constant(0.1, shape = [864], dtype = tf.float32), name = "lyr3_b")

	with tf.name_scope("Layer3"):
		lyr3_opt = tf.nn.relu(tf.add(tf.matmul(flatten_opt, lyr3_w), lyr3_b), name = "reluLyr3")

	with tf.name_scope("paramsAtL4"):
		lyr4_w = tf.Variable(tf.truncated_normal(shape = [864, 432], dtype = tf.float32), name = "lyr4_w")
		lyr4_b = tf.Variable(tf.constant(0.1, shape = [432], dtype = tf.float32), name = "lyr4_b")

	with tf.name_scope("Layer4"):
		lyr4_opt = tf.nn.relu(tf.add(tf.matmul(lyr3_opt, lyr4_w), lyr4_b), name = "reluLyr4")

	with tf.name_scope("paramsAtOptLayer"):
		opt_w = tf.Variable(tf.truncated_normal(shape = [432, N], dtype = tf.float32), name = "opt_w")
		opt_b = tf.Variable(tf.constant(0.1, shape = [N], dtype = tf.float32), name = "opt_b")

	with tf.name_scope("optLayer"):
		resultOp = tf.add(tf.matmul(lyr4_opt, opt_w), opt_b, name = "resultOp")

	with tf.name_scope("crossEntropy"):
		crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = resultOp, labels = input_y), name = "crossEntropy")

	with tf.name_scope("optimizer"):
		optimizer = tf.train.AdamOptimizer(rate).minimize(crossEntropy)

	with tf.name_scope("truthTableBool"):
		truthTableBool = tf.equal(tf.argmax(resultOp, 1), tf.argmax(input_y, 1), name = "truthTableBool")

	with tf.name_scope("truthTableInt"):
		truthTableInt = tf.cast(truthTableBool, tf.int32, name = "truthTableInt")

	with tf.name_scope("prediction"):
		prediction = tf.reduce_sum(truthTableInt, name = "prediction")

	return crossEntropy, optimizer, prediction