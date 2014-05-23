#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import struct
import matplotlib.pyplot as plt

class Converter(object):


	def __init__(self):
		self.train_images = self.read_images('train-images.idx3-ubyte')
		self.test_images = self.read_images('t10k-images.idx3-ubyte')
		self.train_labels = self.read_labels('train-labels.idx1-ubyte')
		self.test_labels = self.read_labels('t10k-labels.idx1-ubyte')


	def read_images(self, filename):
		images = []

		binfile = open(filename , 'rb')
		buf = binfile.read()

		index = 0
		magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index) #read 4unsinged int32
		index += struct.calcsize('>IIII')

		for i in xrange(numImages):
			im = struct.unpack_from('>784B' ,buf, index) #read 784unsigned byte
			index += struct.calcsize('>784B')

			im = np.array(im)
			im = im.reshape(28,28)
			images.append(im)

			# if i == numImages - 1:
			# 	fig = plt.figure()
			# 	plotwindow = fig.add_subplot(111)
			# 	plt.imshow(im , cmap='gray')
			# 	plt.show()

		return np.array(images)


	def read_labels(self, filename):
		labels = []

		binfile = open(filename , 'rb')
		buf = binfile.read()

		index = 0
		magic, numLabels = struct.unpack_from('>II', buf, index) #read 2unsinged int32
		index += struct.calcsize('>II')

		for i in xrange(numLabels):
			im = struct.unpack_from('>B' ,buf, index) #read 1 unsigned byte
			index += struct.calcsize('>B')

			im = np.array(im)
			labels.append(im)

		return np.array(labels)
#endclass Converter

if __name__ == '__main__':
	Converter()