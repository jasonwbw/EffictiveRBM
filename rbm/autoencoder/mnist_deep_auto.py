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
from rbm import RBM, RBMLinear

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
		rbm = RBM(self.converter.dimensionality, 1000)
		rbm.train(self.converter.train_images, max_epochs = 10, batch = 100)
		hidden_probs1 = rbm.hidden_probs
		self.pickle_dumps(rbm.weights, 'l1_w.pkl')
		self.pickle_dumps(rbm.hidden_bias, 'l1_hb.pkl')
		self.pickle_dumps(rbm.visible_bias, 'l1_vb.pkl')
		del rbm
		print "train rbm level 1 end\n"
		
		print "train rbm level 2"
		rbm_l2 = RBM(1000, 500)
		rbm_l2.train(hidden_probs1, max_epochs = 10, batch = 100)
		hidden_probs2 = rbm_l2.hidden_probs
		self.pickle_dumps(rbm_l2.weights, 'l2_w.pkl')
		self.pickle_dumps(rbm_l2.hidden_bias, 'l2_hb.pkl')
		self.pickle_dumps(rbm_l2.visible_bias, 'l2_vb.pkl')
		del rbm_l2
		print "train rbm level 2 end\n"

		print "train rbm level 3"
		rbm_l3 = RBM(500, 250)
		rbm_l3.train(hidden_probs2, max_epochs = 10, batch = 100)
		hidden_probs3 = rbm_l3.hidden_probs
		self.pickle_dumps(rbm_l3.weights, 'l3_w.pkl')
		self.pickle_dumps(rbm_l3.hidden_bias, 'l3_hb.pkl')
		self.pickle_dumps(rbm_l3.visible_bias, 'l3_vb.pkl')
		del rbm_l3
		print "train rbm level 3 end\n"

		print "train rbm level 4"
		rbm_l4 = RBMLinear(250, 30)
		rbm_l4.train(hidden_probs3, max_epochs = 10, batch = 100)
		hidden_top = rbm_l4.hidden_probs
		self.pickle_dumps(rbm_l4.weights, 'l4_w.pkl')
		self.pickle_dumps(rbm_l4.hidden_bias, 'l4_hb.pkl')
		self.pickle_dumps(rbm_l4.visible_bias, 'l4_vb.pkl')
		del rbm_l4
		print "train rbm level 4 end\n"

	def pickle_dumps(self, obj, filename):
		f = open(filename, 'w')
		pickle.dump(obj, f)
		f.close()

if __name__ == '__main__':
	auto = MNISTDeepAuto()
	auto.train()