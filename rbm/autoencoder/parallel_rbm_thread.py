#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a common rbm tools solved by CD-1.

"""

import time
import threading
from copy import deepcopy
from numpy import random, dot, sum, array, exp, zeros, float32 as REAL, concatenate
from scipy.special import expit
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

def one_batch(model, data, \
	initialmomentum, finalmomentum, \
	weight_rate, vbias_rate, hbias_rate, weightcost,\
	epoch, batch_num, max_epochs, lock, isLinear):
	with lock:
		weights = deepcopy(model.weights)
		hidden_bias = deepcopy(model.hidden_bias)
		visible_bias = deepcopy(model.visible_bias)

	pos_hidden_activations = dot(data, weights) + hidden_bias
	if isLinear:
		pos_hidden_probs = pos_hidden_activations
		pos_hidden_states = pos_hidden_probs + random.randn(len(data), len(hidden_bias)) 
	else:
		pos_hidden_probs = expit(pos_hidden_activations)
		pos_hidden_states = pos_hidden_probs > random.randn(len(data), len(hidden_bias))
	posprods = dot(data.T, pos_hidden_probs)
	pos_hidden_act = sum(pos_hidden_probs)
	pos_visible_act = sum(data)

	neg_visible_activations = dot(pos_hidden_states, weights.T) + visible_bias
	neg_visible_probs = expit(neg_visible_activations)
	neg_hidden_activations = dot(neg_visible_probs, weights) + hidden_bias
	if isLinear:
		neg_hidden_probs = neg_hidden_activations
	else:
		neg_hidden_probs = expit(neg_hidden_activations)
	negprods = dot(neg_visible_probs.T, neg_hidden_probs)
	neg_hidden_act = sum(neg_hidden_probs)
	neg_visible_act = sum(neg_visible_probs)
				
	if epoch > 5:
		momentum = finalmomentum
	else:
		momentum = initialmomentum

	add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
	add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
	add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)

	with lock:
		model.error += sum((data - neg_visible_probs) ** 2)
		if epoch == max_epochs - 1:
			if model.hidden_probs == None:
				model.hidden_probs = neg_hidden_probs
			else:
				concatenate((model.hidden_probs, neg_hidden_probs), axis=0)

		model.grad_weight = momentum * model.grad_weight + add_grad_weight
		model.grad_vbias = momentum * model.grad_vbias + add_grad_vbias
		model.grad_hbias = momentum * model.grad_hbias + add_grad_hbias
		model.weights += model.grad_weight
		model.visible_bias += model.grad_vbias
		model.hidden_bias += model.grad_hbias
		
		if batch_num % 10 == 0:
			print 'finish batch', batch_num, time.asctime( time.localtime(time.time()) )

class ParallelRBM(object):

	def __init__(self, num_visible, num_hidden, workers, isLinear = False):
		self.weights = random.randn(num_visible, num_hidden)
		self.hidden_bias = random.randn(1, num_hidden)
		self.visible_bias = random.randn(1, num_visible)

		self.grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)))
		self.grad_hbias = zeros((1, len(self.hidden_bias)))
		self.grad_vbias = zeros((1, len(self.visible_bias)))

		self.hidden_probs = None
		self.workers = workers
		self.isLinear = isLinear

	def train(self, visible_data, \
		max_epochs = 50, batch = 10, \
		initialmomentum = 0.5, finalmomentum = 0.9, \
		weight_rate = 0.001, vbias_rate = 0.001, hbias_rate = 0.001, \
		weightcost = 0.0002):
		
		grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)))
		grad_hbias = zeros((1, len(self.hidden_bias)))
		grad_vbias = zeros((1, len(self.visible_bias)))
		jobs = Queue(maxsize=2 * self.workers) 
		lock = threading.Lock()
		
		def worker_train():
			"""Train the model, lifting lists of data from the jobs queue."""
			while True:
				job = jobs.get()
				if job is None:
					break
				one_batch(job[0], job[1], initialmomentum, finalmomentum, \
					weight_rate, vbias_rate, hbias_rate, weightcost,\
					job[2], job[3], max_epochs, lock, self.isLinear)
		#end_worker_train

		for epoch in xrange(max_epochs):
			print "epoch %d" % (epoch)
			self.error = 0.
			workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
			for thread in workers:
				# make interrupting the process with ctrl+c easier
				thread.daemon = True
				thread.start()
			for start in xrange(batch):
				data = visible_data[start::batch]
				jobs.put( (self, data, epoch, start) )
			for _ in xrange(self.workers):
				jobs.put(None) # give the workers heads up that they can finish -- no more work!  
			for thread in workers:
				thread.join()
			print "epoch %d, error %d\n" % (epoch, self.error)
#endclass ParallelRBM

if __name__ == '__main__':
	r = RBM(num_visible = 6, num_hidden = 2)
	training_data = array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
	r.train(training_data, batch = 2)
	print 'weights', r.weights
	print 'hidden bias', r.hidden_bias
	print 'visible bias', r.visible_bias