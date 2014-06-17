#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : Jasonwbw@yahoo.com

"""
This is a common rbm tools solved by CD-1.

"""

import pickle
from converter import Converter
from numpy import insert, random, dot, sum, array, exp, zeros, float32 as REAL, concatenate
from scipy.special import expit

def pickle_load(filename):
	f = open(filename, 'r')
	res = pickle.load(f)
	f.close()
	return res

def load_weights():
	we1 = pickle_load("l1_w.pkl")
	we2 = pickle_load("l2_w.pkl")
	we3 = pickle_load("l3_w.pkl")
	we4 = pickle_load("l4_w.pkl")
	return we1, we2, we3, we4

def load_hidden_bias():
	hb1 = pickle_load("l1_hb.pkl")
	hb2 = pickle_load("l2_hb.pkl")
	hb3 = pickle_load("l3_hb.pkl")
	hb4 = pickle_load("l4_hb.pkl")
	return hb1, hb2, hb3, hb4

def load_visible_bias():
	vb1 = pickle_load("l1_vb.pkl")
	vb2 = pickle_load("l2_vb.pkl")
	vb3 = pickle_load("l3_vb.pkl")
	vb4 = pickle_load("l4_vb.pkl")
	return vb1, vb2, vb3, vb4

def compute_error(data, dimensionality, batch):
	error = 0.
	for start in xrange(batch): 
		if start % 10 == 0:
			print 'batch', start
		data = insert(data[start::batch], dimensionality, 1, axis = 1) 
		w1probs = insert(expit(dot(data, w1)), 1000, 1, axis = 1)
		w2probs = insert(expit(dot(w1probs, w2)), 500, 1, axis = 1)
		w3probs = insert(expit(dot(w2probs, w3)), 250, 1, axis = 1)
		w4probs = insert(dot(w3probs, w4), 30, 1, axis = 1)
		w5probs = insert(expit(dot(w4probs, w5)), 250, 1, axis = 1)
		w6probs = insert(expit(dot(w5probs, w6)), 500, 1, axis = 1)
		w7probs = insert(expit(dot(w6probs, w7)), 1000, 1, axis = 1)
		dataout = expit(dot(w7probs, w8))
		error += 1. / len(data) * sum((data[start::batch] - dataout)  ** 2)
	return error / batch

def handle():
	print 'load data'
	global we1, we2, we3, we4
	global hb1, hb2, hb3, hb4
	global vb1, vb2, vb3, vb4
	we1, we2, we3, we4 = load_weights()
	hb1, hb2, hb3, hb4 = load_hidden_bias()
	vb1, vb2, vb3, vb4 = load_visible_bias()	
	
	print '\nrebuild data'
	w1 = concatenate((we1, hb1), axis=0)
	w2 = concatenate((we2, hb2), axis=0)
	w3 = concatenate((we3, hb3), axis=0)
	w4 = concatenate((we4, hb4), axis=0)
	w5 = concatenate((we4.T, vb4), axis=0)
	w6 = concatenate((we3.T, vb3), axis=0)
	w7 = concatenate((we2.T, vb2), axis=0)
	w8 = concatenate((we1.T, vb1), axis=0)

	print '\nbegin converting data into memory'
	converter = Converter()
	batch = 100

	print '\nstart train error compute'
	train_error = compute_error(converter.train_images, converter.dimensionality, batch)
	print train_error

	print '\nstart test error compute'
	test_error = compute_error(converter.test_images, converter.dimensionality, batch)
	print test_error


if __name__ == '__main__':
	handle()