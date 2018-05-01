# Programmer : Dev Bishnoi

# This file contains multiple function to preprocess datasets provided. Use appropriate function according to your requirement.
# If you are running first method then make comment to another one.

import csv
import numpy as np
import matplotlib.pyplot as plt



#######################################################################################################################
#before using this script please have a look on this file.
file_read = open("F:/............/emnist-balanced-train.csv", 'r')
file_write = open("F:/............/emnist_train.csv", 'w', newline='')
reader = csv.reader(file_read, delimiter=',')
writer = csv.writer(file_write, delimiter=',')

# It extract digits and capital alphabet letters from emnist-balanced-train.csv into emnist_train.csv.
def extract():
	for row in reader:
		label = int(row[0])
		if(label <= 35):
			writer.writerow(row)
extract()

########################################################################################################################
# Just to visualize data, weather they are correctly processed or not.
file_write = open("F:/............/emnist_train.csv", 'w', newline='')
reader = csv.reader(file_read, delimiter=',')
def showData():
	cnt = 0
	for row in reader:
		cnt += 1
		idx = int(row[0])
		label = np.zeros([36])
		label[idx] = 1
		img = row[1:]
		image = [int(x) for x in img]
		image_file = np.reshape(image, [28, 28])
		print(idx)
		plt.imshow(image_file)
		plt.show()
		if(cnt == 10):
			break
showData()



###############################################################################################################################

# Flip vertically and then rotate by 90 degree anti-clock-wise.
# Image is being loaded from emnist_train.csv and then processed (flipped and rotated ) image is written to train.csv
fread = open("F:/............/emnist_train.csv", 'r')
fwrite = open("F:/............/train.csv", 'w', newline='')
reader = csv.reader(fread, delimiter=',')
writer = csv.writer(fwrite, delimiter = ',')
def preprocess():
	for row in reader:
		img = row[1:]
		image = np.reshape(img, [28, 28])
		for i in range(28):
			for j in range(28):
				if(i > j):
					item = image[i][j]
					image[i][j] = image[j][i]
					image[j][i] = item
		image = np.reshape(image, [28 * 28])
		l = len(image)
		for i in range(l):
			row[i+1] = image[i]
		writer.writerow(row)
preprocess()