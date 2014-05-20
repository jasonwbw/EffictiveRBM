#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a common rbm tools solved by CD-k.

"""

from numpy import random, dot, sum, array, exp, zeros


class RBM(object):


	def __init__(self, num_visible, num_hidden):
		self.weights = random.randn(num_visible, num_hidden)
		self.hidden_bias = random.randn(1, num_hidden)
		self.visible_bias = random.randn(1, num_visible)


	def train(self, visible_data, \
		max_epochs = 50, batch = 10, \
		initialmomentum = 0.5, finalmomentum = 0.9, \
		weight_rate = 0.1, vbias_rate = 0.1, hbias_rate = 0.1, \
		weightcost = 0.0002):
		grad_weight = zeros((len(self.visible_bias), len(self.hidden_bias)))
		grad_hbias = zeros((1, len(self.hidden_bias)))
		grad_vbias = zeros((1, len(self.visible_bias)))
		error = 0
		for epoch in xrange(max_epochs):
			for start in xrange(batch):
				print "epoch %d, batch %d" % (epoch, start + 1)
				data = visible_data[start::batch]
				pos_hidden_activations = dot(data, self.weights) + self.hidden_bias
				pos_hidden_probs = self._sigmoid(pos_hidden_activations)
				pos_hidden_states = pos_hidden_probs > random.randn(len(data), len(self.hidden_bias) + 1)
				posprods = dot(data.T, pos_hidden_probs)
				pos_hidden_act = sum(pos_hidden_probs)
				pos_visible_act = sum(data)

				neg_visible_activations = dot(pos_hidden_states, self.weights.T) + self.visible_bias
				neg_visible_probs = self._sigmoid(neg_visible_activations)
				neg_hidden_activations = dot(neg_visible_probs, self.weights) + self.hidden_bias
				neg_hidden_probs = self._sigmoid(neg_hidden_activations)
				negprods = dot(neg_visible_probs.T, neg_hidden_probs)
				neg_hidden_act = sum(neg_hidden_probs)
				neg_visible_act = sum(neg_visible_probs)
				error += sum((data - neg_visible_probs) ** 2)
				
				if epoch > 5:
					momentum = finalmomentum
				else:
					momentum = initialmomentum
				grad_weight = momentum * grad_weight + weight_rate * ((posprods - negprods) / len(data) - weightcost * self.weights)
				grad_vbias = momentum * grad_vbias + vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
				grad_hbias = momentum * grad_hbias + hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)
				
				self.weights += grad_weight
				self.visible_bias += grad_vbias
				self.hidden_bias += grad_hbias
				print "epoch %d, error %d" % (epoch, error)


	def _sigmoid(self, x):
		return 1.0 / (1 + exp(-x))
#endclass RBM

if __name__ == '__main__':
	r = RBM(num_visible = 6, num_hidden = 2)
	training_data = array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0],[0,0,1,1,1,1]])
	r.train(training_data, batch = 1)
	print(r.weights)
