#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : @Jason_wbw

"""
This program pertrains a deep autoencoder for MNIST dataset
You cat set the maxinum number of epochs for pertraining each layer 
    and you can set the architectrue of the multiplayer nets.
"""

from converter import Converter
from parallel_rbm_thread import ParallelRBM

import pickle

class MNISTDeepAuto(object):

	def __init__(self, batch_num = 100):
		self._load_data()

	def _load_data(self):
		print "begin converting data into memory"
		self.converter = Converter()
		print "converting end\n"

	def train(self):
		print "train rbm level 1"
		rbm = ParallelRBM(self.converter.dimensionality, 1000, 10)
		rbm.train(self.converter.train_images, max_epochs = 10, batch = 100)
		hidden_probs1 = rbm.hidden_probs
		self.pickle_dumps(rbm.weights, 'l1_w_p.pkl')
		self.pickle_dumps(rbm.hidden_bias, 'l1_hb_p.pkl')
		self.pickle_dumps(rbm.visible_bias, 'l1_vb_p.pkl')
		del rbm
		del self.converter
		print "train rbm level 1 end\n"
		
		print "train rbm level 2"
		rbm_l2 = ParallelRBM(1000, 500, 10)
		rbm_l2.train(hidden_probs1, max_epochs = 10, batch = 100)
		hidden_probs2 = rbm_l2.hidden_probs
		self.pickle_dumps(rbm_l2.weights, 'l2_w_p.pkl')
		self.pickle_dumps(rbm_l2.hidden_bias, 'l2_hb_p.pkl')
		self.pickle_dumps(rbm_l2.visible_bias, 'l2_vb_p.pkl')
		del rbm_l2
		del hidden_probs1
		print "train rbm level 2 end\n"

		print "train rbm level 3"
		rbm_l3 = ParallelRBM(500, 250, 10)
		rbm_l3.train(hidden_probs2, max_epochs = 10, batch = 100)
		hidden_probs3 = rbm_l3.hidden_probs
		self.pickle_dumps(rbm_l3.weights, 'l3_w_p.pkl')
		self.pickle_dumps(rbm_l3.hidden_bias, 'l3_hb_p.pkl')
		self.pickle_dumps(rbm_l3.visible_bias, 'l3_vb_p.pkl')
		del rbm_l3
		del hidden_probs2
		print "train rbm level 3 end\n"

		print "train rbm level 4"
		rbm_l4 = ParallelRBM(250, 30, 10, isLinear = True)
		rbm_l4.train(hidden_probs3, max_epochs = 10, batch = 100)
		hidden_top = rbm_l4.hidden_probs
		self.pickle_dumps(rbm_l4.weights, 'l4_w_p.pkl')
		self.pickle_dumps(rbm_l4.hidden_bias, 'l4_hb_p.pkl')
		self.pickle_dumps(rbm_l4.visible_bias, 'l4_vb_p.pkl')
		del rbm_l4
		del hidden_probs3
		print "train rbm level 4 end\n"

	def pickle_dumps(self, obj, filename):
		f = open(filename, 'w')
		pickle.dump(obj, f)
		f.close()

if __name__ == '__main__':
	auto = MNISTDeepAuto()
	auto.train()