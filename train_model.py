# Programmer : Dev Bishnoi

# This module trains model constructed in model module.

import tensorflow as tf
import numpy as np
import model as m
import feed_data
import os
import csv
import pickle
import setBack as sb


batch_size = 30
show_steps = 50
logdir = "./tmp/data/1"
file_handle = open("C:/Users/Devilal/Documents/Machine Learning/Data/emnist-0A/emnist_train.csv", "r")
reader = csv.reader(file_handle, delimiter = ',')


file_handle_test = open("C:/Users/Devilal/Documents/Machine Learning/Data/emnist-0A/emnist_test.csv", "r")
reader_test = csv.reader(file_handle_test, delimiter = ',')


def getLength(rdr):
	cnt = 0
	for row in rdr:
		cnt += 1
	return cnt

train_size = getLength(reader)
test_size = getLength(reader_test)

def set_file_handle_to_beginning(fh):
	fh.seek(0)
	return

def train_model():

	writer = tf.summary.FileWriter(logdir) ###################### Add This writter to file
	pretrained = os.path.isdir('store')
	if(pretrained == False):
		#place-holders
		stb = sb.Setback(1, 1)
		tf.reset_default_graph()
		with tf.name_scope("inputs"):
			input_x = tf.placeholder(tf.float32, shape = [None, m.width * m.height], name = "input_x")
			input_y = tf.placeholder(tf.int32, shape = [None, m.N], name = "input_y")

		#build model
		crossEntropy, optimizer, prediction = m.model(input_x, input_y)

		saver = tf.train.Saver()
		tf.add_to_collection("input_x", input_x)
		tf.add_to_collection("input_y", input_y)
		tf.add_to_collection("crossEntropy", crossEntropy)
		tf.add_to_collection("optimizer", optimizer)
		tf.add_to_collection("prediction", prediction)

	totalCost = tf.placeholder(tf.float32)
	trainAcc = tf.placeholder(tf.float32)
	tf.summary.scalar('Cost', totalCost)
	tf.summary.scalar('Train Accuracy', trainAcc)
	merge_op = tf.summary.merge_all()

	with tf.Session() as sess:

		if(pretrained == False):
			sess.run(tf.global_variables_initializer())
			writer.add_graph(sess.graph)
		else:
			saver = tf.train.import_meta_graph("store/my_model.meta")
			saver.restore(sess, tf.train.latest_checkpoint("store/"))
			input_x = tf.get_collection("input_x")[0]
			input_y = tf.get_collection("input_y")[0]
			crossEntropy = tf.get_collection("crossEntropy")[0]
			optimizer = tf.get_collection("optimizer")[0]
			prediction = tf.get_collection("prediction")[0]
			with open("store/info.obj", "rb") as f:
				stb = pickle.load(f)
		# In each ep whole trained data is explored.
		epochs = int(input("Enter the number of epochs:"))
		iterations = stb.fin_itr
		prev_ep = stb.eps
		mx_acc = 0
		drop_cnt = 0
		bflag = False
		for ep in range(epochs):

			set_file_handle_to_beginning(file_handle)
			total_cost = 0
			total_correct_pred = 0
			total_preds = 0
			# In each i a batch of size batch_size data is processed.
			nit = int(train_size/batch_size)
			for i in range(nit):
				batch_x, batch_y = feed_data.fetchData(reader, batch_size)
				_, cost, correct_pred = sess.run([optimizer, crossEntropy, prediction], feed_dict = {input_x : batch_x, input_y : batch_y})
				total_cost += cost
				total_correct_pred += correct_pred
				total_preds += batch_size

				if((i + 1) % show_steps == 0):
					train_accuracy = total_correct_pred / total_preds
					print("step: ", ep + prev_ep, " ", iterations, " cost: ", total_cost, ", TrainAcc: ", total_correct_pred, "/", total_preds)
					merge_summary = sess.run(merge_op, feed_dict={totalCost:total_cost, trainAcc:train_accuracy})
					writer.add_summary(merge_summary, iterations)
					if(mx_acc >= total_correct_pred):
						drop_cnt += 1;
					else:
						drop_cnt = 0
						mx_acc = total_correct_pred
					if(drop_cnt == 15):
						print("Model Accuracy is not increasing since last 15 iterations. So we are interrupting training process.")
						bflag = True
						break
					total_cost = 0
					total_correct_pred = 0
					total_preds = 0
					iterations += 1

			total_preds = 0
			total_correct_pred = 0
			set_file_handle_to_beginning(file_handle_test)
			for i in range(int(test_size / batch_size)):
				batch_x, batch_y = feed_data.fetchData(reader_test, batch_size)
				correct_pred = sess.run(prediction, feed_dict={input_x:batch_x, input_y:batch_y})
				total_correct_pred += correct_pred
				total_preds += batch_size
			print("Test Accuracy: ", total_correct_pred, "/", total_preds)
			if(bflag):
				break

		saver.save(sess, "store/my_model")
		stb.fin_itr = iterations + 1
		stb.eps = prev_ep + ep + 1
		with open("store/info.obj", 'wb') as f:
			pickle.dump(stb, f)
	writer.close()
train_model()
file_handle.close()