#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : Jasonwbw@yahoo.com

from abc import ABCMeta, abstractmethod

class BatchIterator(object):

	__metaclass__ = ABCMeta

	def __init__(self, batch):
		self.batch = batch

	@abstractmethod
	def next_batch(self):
		pass

	@abstractmethod
	def back2start(self):
		pass

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
		self.current_batch = -1

	def next_batch(self):
		if self.current_batch >= self.batch - 1:
			return None
		self.current_batch += 1
		return self.data[self.current_batch : : self.batch]

	def back2start(self):
		self.current_batch = -1
#endclass DefaultBatchIterator

if __name__ == '__main__':
	converter = Converter()