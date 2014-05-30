#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a common rbm tools solved by CD-1.

"""

import time
from numpy import random, dot, sum, array, exp, zeros, float32 as REAL, concatenate
from scipy.special import expit

class RBM(object):

	def __init__(self, num_visible, num_hidden, isLinear = False):
		self.weights = random.randn(num_visible, num_hidden)
		self.hidden_bias = random.randn(1, num_hidden)
		self.visible_bias = random.randn(1, num_visible)
		self.hidden_probs = None
		self.isLinear = isLinear

	def train(self, visible_data, \
		max_epochs = 50, batch = 10, \
		initialmomentum = 0.5, finalmomentum = 0.9, \
		weight_rate = 0.001, vbias_rate = 0.001, hbias_rate = 0.001, \
		weightcost = 0.0002):
		grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)))
		grad_hbias = zeros((1, len(self.hidden_bias)))
		grad_vbias = zeros((1, len(self.visible_bias)))
		for epoch in xrange(max_epochs):
			error = 0.
			for start in xrange(batch):
				if start % 10 == 0:
					print "epoch %d, batch %d" % (epoch, start + 1), time.asctime( time.localtime(time.time()) )
				data = visible_data[start::batch]
				pos_hidden_activations = dot(data, self.weights) + self.hidden_bias
				if self.isLinear:
					pos_hidden_probs = pos_hidden_activations
					pos_hidden_states = pos_hidden_probs + random.randn(len(data), len(self.hidden_bias))
				else:
					pos_hidden_probs = expit(pos_hidden_activations)
					pos_hidden_states = pos_hidden_probs > random.randn(len(data), len(self.hidden_bias))
				posprods = dot(data.T, pos_hidden_probs)
				pos_hidden_act = sum(pos_hidden_probs)
				pos_visible_act = sum(data)

				neg_visible_activations = dot(pos_hidden_states, self.weights.T) + self.visible_bias
				neg_visible_probs = expit(neg_visible_activations)
				neg_hidden_activations = dot(neg_visible_probs, self.weights) + self.hidden_bias
				if self.isLinear:
					neg_hidden_probs = neg_hidden_activations
				else:
					neg_hidden_probs = expit(neg_hidden_activations)
				negprods = dot(neg_visible_probs.T, neg_hidden_probs)
				neg_hidden_act = sum(neg_hidden_probs)
				neg_visible_act = sum(neg_visible_probs)
				error += sum((data - neg_visible_probs) ** 2)
				
				if epoch > 5:
					momentum = finalmomentum
				else:
					momentum = initialmomentum
				
				if epoch == max_epochs - 1:
					if self.hidden_probs == None:
						self.hidden_probs = neg_hidden_probs
					else:
						self.hidden_probs = concatenate((self.hidden_probs, neg_hidden_probs), axis=0)

				grad_weight = momentum * grad_weight + weight_rate * ((posprods - negprods) / len(data) - weightcost * self.weights)
				grad_vbias = momentum * grad_vbias + vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
				grad_hbias = momentum * grad_hbias + hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)
				
				self.weights += grad_weight
				self.visible_bias += grad_vbias
				self.hidden_bias += grad_hbias
			print "epoch %d, error %d\n" % (epoch, error)
#endclass RBM

if __name__ == '__main__':
	r = RBM(num_visible = 6, num_hidden = 2)
	training_data = array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
	r.train(training_data, batch = 2)
	print 'weights', r.weights
	print 'hidden bias', r.hidden_bias
	print 'visible bias', r.visible_bias