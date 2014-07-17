#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : Jasonwbw@yahoo.com

"""
This is a common multi process rbm tools solved by CD-1.

References :
[Geoffrey Hinton] A Practical Guide to Training Restricted Boltzmann Machines
[Geoffrey Hinton] Reducing the Dimensionality of data with neural networks
"""

import time
import multiprocessing as mp
import joblib
from copy import deepcopy
from numpy import random, dot, sum, array, exp, zeros, concatenate, float32 as REAL
from scipy.special import expit

try:
	import pyximport
	import numpy as np
	pyximport.install(setup_args={"include_dirs":np.get_include()},
		reload_support=True)
	from worker_inner import fast_worker as cworker
	def worker((data,\
		weights, hidden_bias, visible_bias,\
		weight_rate, vbias_rate, hbias_rate, weightcost, isLinear, batch_num)):
		return cworker(data,\
			weights, hidden_bias, visible_bias,\
			weight_rate, vbias_rate, hbias_rate, weightcost, isLinear, batch_num)
except ImportError:
	def worker((data,\
		weights, hidden_bias, visible_bias,\
		weight_rate, vbias_rate, hbias_rate, weightcost, isLinear, batch_num)):
		'''
		One process's job that compute the added gradient

		Args:
		    data : data for this batch
            weights : weights in rbm model
            hidden_bias : hidden bias in rbm model
            visible_bias : visible bias in rbm model
            weight_rate : learning for weight
            vbias_rate : learning for visible bias
            hbias_rate : learning for hidden bias
            weightcost : weight cost in rbm model
            isLinear : is linear rbm model or not
            batch_num : number for this batch

        Returns:
            tuple of error, added gradient for weight, added gradient for visible bias, added gradient for hidden bias
        '''
		pos_hidden_activations = dot(data, weights) + hidden_bias
		if isLinear:
			pos_hidden_probs = pos_hidden_activations
			pos_hidden_states = pos_hidden_probs + random.randn(len(data), len(hidden_bias)).astype(REAL)
		else:
			pos_hidden_probs = expit(pos_hidden_activations)
			pos_hidden_states = pos_hidden_probs > random.randn(len(data), len(hidden_bias)).astype(REAL)
		posprods = dot(data.T, pos_hidden_probs)
		pos_hidden_act = sum(pos_hidden_probs, axis = 0)
		pos_visible_act = sum(data, axis = 0)
		neg_visible_activations = dot(pos_hidden_states, weights.T) + visible_bias
		neg_visible_probs = expit(neg_visible_activations)
		neg_hidden_activations = dot(neg_visible_probs, weights) + hidden_bias
		if isLinear:
			neg_hidden_probs = neg_hidden_activations
		else:
			neg_hidden_probs = expit(neg_hidden_activations)
		negprods = dot(neg_visible_probs.T, neg_hidden_probs)
		neg_hidden_act = sum(neg_hidden_probs, axis = 0)
		neg_visible_act = sum(neg_visible_probs, axis = 0)
		add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
		add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
		add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)
		error = sum((data - neg_visible_probs) ** 2)
		if batch_num % 10 == 0:
			print 'finish batch compute', batch_num, time.asctime(time.localtime(time.time())), '. Added error', error
		return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

class ParallelRBM(object):

	'''
	RBM model with multi process learning

	Attributes:
	    weights : weights in rbm model
        hidden_bias : hidden bias in rbm model
        visible_bias : visible bias in rbm model 
	    hidden_probs : hidden values that predicted
	    workers : number of process to use
	    isLinear : is linear rbm or not
	    _grad_weight : tmp gradient for weight
	    _grad_hbias ： tmp gradient for hidden bias
	    _grad_vbias : tmp gradient for visible bias
	'''

	def __init__(self, num_visible, num_hidden, workers, model_folder = './model', isLinear = False):
		'''
		Init attributes by given params

		Args:
		    num_visible : number of visible node
		    num_hidden : number of hidden node
		    workers : number of process to use
		    isLinear : is linear rbm or not
		'''
		self.weights = random.randn(num_visible, num_hidden).astype(REAL)
		self.hidden_bias = random.randn(1, num_hidden).astype(REAL)
		self.visible_bias = random.randn(1, num_visible).astype(REAL)

		self._grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)), dtype=REAL)
		self._grad_hbias = zeros((1, len(self.hidden_bias)), dtype=REAL)
		self._grad_vbias = zeros((1, len(self.visible_bias)), dtype=REAL)

		self.hidden_probs = None
		self.workers = workers
		self.isLinear = isLinear
		self.model_folder = model_folder
		self.last_epoch = -1

	def train(self, batch_iter, \
		max_epochs = 50, batch = 10, \
		initialmomentum = 0.5, finalmomentum = 0.9, \
		weight_rate = 0.001, vbias_rate = 0.001, hbias_rate = 0.001, \
		weightcost = 0.0002, rate_m=1.0):
		'''
		Train the rbm model for the data and given learning params

		Args:
		    visible_data : one-dimensional input array with size = num_visible in init method 
		    max_epochs : epoch to learning
		    batch : batch num in one epoch
		    initialmomentum : initial momentum value
		    finalmomentum : final momentum value that epoch number > 5
		    weight_rate : learning rate for weight
		    vbias_rate : learning rate for visible bias
		    hbias_rate : learning rate for hidden bias
		    weightcost : weightcost for gradient update
		'''
		if batch < self.workers or batch % self.workers != 0:
			print 'choose batch of multiple workers will be more efficient, current workers is', self.workers
			return 
		self.max_epochs = max_epochs
		self._grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)), dtype=REAL)
		self._grad_hbias = zeros((1, len(self.hidden_bias)), dtype=REAL)
		self._grad_vbias = zeros((1, len(self.visible_bias)), dtype=REAL)
		for epoch in xrange(max_epochs):
			print "epoch %d" % (epoch)
			if epoch > 5:
				momentum = finalmomentum
			else:
				momentum = initialmomentum
			self.error = 0.
			#init wokers twice as self.workers, and add self.workers job after one batch self.workers end
			pool = mp.Pool(self.workers)
			rel1 = pool.imap_unordered(worker, [(batch_iter.next(), self.weights, self.hidden_bias, self.visible_bias,\
				weight_rate, vbias_rate, hbias_rate, weightcost, self.isLinear, b) for b in xrange(min(2 * self.workers, batch))])
			for b in xrange(self.workers):
				weight_rate *= rate_m
				hbias_rate *= rate_m
				vbias_rate *= rate_m
				self.update(rel1.next(), epoch, momentum)
			for turn in xrange(batch / self.workers - 1):
				if turn != batch / self.workers - 2:
					rel_tmp = pool.imap_unordered(worker, [(batch_iter.next(), self.weights, self.hidden_bias, self.visible_bias,\
						weight_rate, vbias_rate, hbias_rate, weightcost, self.isLinear, b) for b in xrange((turn + 2) * self.workers, (turn + 3) * self.workers)])
					if turn % 2 == 0:
						rel2 = rel_tmp
					else:
						rel1 = rel_tmp
				for b in xrange(self.workers):
					if turn % 2 == 0:
						weight_rate *= rate_m
						hbias_rate *= rate_m
						vbias_rate *= rate_m
						self.update(rel1.next(), epoch, momentum)
					else:
						weight_rate *= rate_m
						hbias_rate *= rate_m
						vbias_rate *= rate_m
						self.update(rel2.next(), epoch, momentum)
			batch_iter.back2start()
			print "epoch %d, error %d\n" % (epoch, self.error)
			self.save(epoch = epoch)

	def update(self, (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs), epoch, momentum):
		'''
		Update the gradient and attributes

		Args:
		    args0 tuple - error : current error in this epoch
		    args0 tuple - add_grad_weight : added gradient for weight in this batch
		    args0 tuple - add_grad_vbias : added gradient for visible bias in this batch
		    args0 tuple - add_grad_hbias : added gradient for hidden bias in this batch
		    args0 tuple - neg_hidden_probs : neg_hidden_probs that to save
		'''
		self.error += error
		
		if self.hidden_probs == None or len(self.hidden_probs) == 0 or epoch != self.last_epoch:
			self.last_epoch = epoch
			self.hidden_probs = neg_hidden_probs
		else:
			concatenate((self.hidden_probs, neg_hidden_probs), axis=0)
		
		self._grad_weight = momentum * self._grad_weight + add_grad_weight
		self._grad_vbias = momentum * self._grad_vbias + add_grad_vbias
		self._grad_hbias = momentum * self._grad_hbias + add_grad_hbias
		
		self.weights += self._grad_weight
		self.visible_bias += self._grad_vbias
		self.hidden_bias += self._grad_hbias

	def dimension_reduction(self, x):
		pos_hidden_activations = dot(x, self.weights) + self.hidden_bias
		if self.isLinear:
			return pos_hidden_activations
		else:
			return expit(pos_hidden_activations)

	def save(self, epoch = None):
		if epoch == None:
			joblib.dump(self, self.model_folder) + '/model.pkl'
		else:
			joblib.dump(self, self.model_folder + '/' + str(epoch) + '/model.pkl')

	@classmethod
	def load(cls, model_folder):
		return joblib.load(model_folder)
#endclass ParallelRBM