# Programmer : Dev Bishnoi

import csv
import numpy as np

def fetchData(reader, batch_size):
	cnt = 1
	images = []
	labels = []
	for row in reader:
		idx = int(row[0])
		label = np.zeros([36])
		label[idx] = 1
		img = row[1:]
		image = [np.float32(x) for x in img]
		images.append(image)
		labels.append(label)
		if(cnt == batch_size):
			break
		cnt += 1
	return images, labels