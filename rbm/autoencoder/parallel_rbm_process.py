#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a common rbm tools solved by CD-1.

"""

import time
import multiprocessing as mp
from copy import deepcopy
from numpy import random, dot, sum, array, exp, zeros, float32 as REAL, concatenate
from scipy.special import expit

def worker((data,\
	weights, hidden_bias, visible_bias,\
	weight_rate, vbias_rate, hbias_rate, weightcost, isLinear, batch_num)):

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

	add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
	add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
	add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)

	error = sum((data - neg_visible_probs) ** 2)

	if batch_num % 10 == 0:
		print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )

	return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

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
		self.max_epochs = max_epochs
		self.grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)))
		self.grad_hbias = zeros((1, len(self.hidden_bias)))
		self.grad_vbias = zeros((1, len(self.visible_bias)))
		for epoch in xrange(max_epochs):
			print "epoch %d" % (epoch)
			if epoch > 5:
				momentum = finalmomentum
			else:
				momentum = initialmomentum
			self.error = 0.
			pool = mp.Pool(self.workers)
			rel = pool.imap_unordered(worker, [(visible_data[b::batch], self.weights, self.hidden_bias, self.visible_bias,\
				weight_rate, vbias_rate, hbias_rate, weightcost, self.isLinear, b) for b in xrange(batch)])
			for b in xrange(batch):
				self.update(rel.next(), epoch, momentum)
			print "epoch %d, error %d\n" % (epoch, self.error)

	def update(self, (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs), epoch, momentum):
		self.error += error
		
		if epoch == self.max_epochs - 1:
			if len(self.hidden_probs) == 0:
				self.hidden_probs = neg_hidden_probs
			else:
				concatenate((self.hidden_probs, neg_hidden_probs), axis=0)
		
		self.grad_weight = momentum * self.grad_weight + add_grad_weight
		self.grad_vbias = momentum * self.grad_vbias + add_grad_vbias
		self.grad_hbias = momentum * self.grad_hbias + add_grad_hbias
		
		self.weights += self.grad_weight
		self.visible_bias += self.grad_vbias
		self.hidden_bias += self.grad_hbias
#endclass ParallelRBM