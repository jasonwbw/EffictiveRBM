#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : Jasonwbw@yahoo.com

from abc import ABCMeta, abstractmethod
import numpy as np

class BatchIterator(object):

	__metaclass__ = ABCMeta

	def __init__(self, batch):
		self.batch = batch
		self.current_batch = -1

	@abstractmethod
	def next_batch(self):
		pass

	def back2start(self):
		self.current_batch = -1

	def next(self):
		next_batch = self.next_batch()
		if next_batch == None:
			raise StopIteration
		return next_batch

	def __iter__(self):
		return self
#endclass BatchIterator

class DefaultBatchIterator(BatchIterator):

	def __init__(self, batch, data):
		BatchIterator.__init__(self, batch)
		self.data = data

	def next_batch(self):
		if self.current_batch >= self.batch - 1:
			return None
		self.current_batch += 1
		return self.data[self.current_batch : : self.batch]
#endclass DefaultBatchIterator

class FileBatchIterator(BatchIterator):

	def _file_length(self, filename):
		count = 0
		with open(filename, 'r') as fp:
			for line in fp:
				count += 1
		return count

	def __init__(self, batch, filename):
		BatchIterator.__init__(self, batch)
		self.filename = filename
		self.file_length = self._file_length(filename)
		self._f = open(filename, 'r')
		if self.file_length % batch == 0:
			self.one_batch_line = self.file_length / batch
		else:
			self.one_batch_line = self.file_length / batch + 1

	def next_batch(self):
		if self.current_batch >= self.batch - 1:
			return None
		self.current_batch += 1
		data = []
		for i in xrange(min(self.one_batch_line, self.file_length - self.current_batch * self.one_batch_line)):
			data.append(self._f.readline().strip())
		return np.array(data)

	def back2start(self):
		BatchIterator.back2start(self)
		self._f.close()
		self._f = open(self.filename, 'r')
#endclass FileBatchIterator

if __name__ == '__main__':
	fbi = FileBatchIterator(5, 'backprop.py')
	print len(fbi.next())
	print len(fbi.next())
	print len(fbi.next())
	print len(fbi.next())
	print len(fbi.next())