#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : @Jason_wbw

"""
Converter MNIST data to np.array

MNIST files are available at http://yann.lecun.com/exdb/mnist/
Before using this program you first need to download files:
train-images.idx3-ubyte.gz : train image data
t10k-images.idx3-ubyte.gz  : test image data
train-labels.idx1-ubyte.gz : train labels data
t10k-labels.idx1-ubyte.gz  : test labels data
and gunzip them.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt

class Converter(object):

	def __init__(self):
		#self.test_read('t10k-images.idx3-ubyte')
		train_images = self.read_images('train-images.idx3-ubyte')
		test_images = self.read_images('t10k-images.idx3-ubyte')
		self.train_labels = self.read_labels('train-labels.idx1-ubyte')
		self.test_labels = self.read_labels('t10k-labels.idx1-ubyte')
		self.train_images, self.test_images = self.simple_scale(train_images, test_images)
		self.total_train_data = len(self.train_labels)
		self.dimensionality = 28 * 28

	def simple_scale(self, data0, data1):
		return data0 / 255., data1 / 255.

	def test_read(self, filename):
		images = []

		binfile = open(filename , 'rb')
		buf = binfile.read()

		index = 0
		magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index) #read 4unsinged int32
		index += struct.calcsize('>IIII')

		im = struct.unpack_from('>%dB' % (28*28), buf, index) #read 784unsigned byte
		index += struct.calcsize('>%dB' % (28*28))

		im = np.array(im)
		im = im.reshape(28, 28) 
		fig = plt.figure()
		plotwindow = fig.add_subplot(111)
		plt.imshow(im , cmap='gray')
		plt.show()

	def read_images(self, filename):
		images = []

		binfile = open(filename , 'rb')
		buf = binfile.read()

		index = 0
		magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index) #read 4 unsinged int32
		index += struct.calcsize('>IIII')

		for i in xrange(numImages):
			im = struct.unpack_from('>%dB' % (28*28), buf, index) #read 784unsigned byte
			index += struct.calcsize('>%dB' % (28*28))

			im = np.array(im)
			images.append(im)

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
	converter = Converter()