#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This is a common rbm tools solved by CD-1.

"""

import pickle
from converter import Converter
from numpy import insert, random, dot, sum, array, exp, zeros, float32 as REAL, concatenate

def sigmoid(X):
	vec_sigmoid = lambda X : array([oneelem_sigmoid(x) for x in X], dtype = REAL) 
	sigmoid = lambda X : array([vec_sigmoid(x) for x in X], dtype = REAL) 
	return sigmoid(X)

def oneelem_sigmoid(x):
	if x >= 0:
		value = 1. / (1 + exp(-x))
	else:
		tmp = exp(x)
		value = tmp / (1 + tmp)
	return value

def pickle_load(filename):
	f = open(filename, 'r')
	res = pickle.load(f)
	f.close()
	return res

def handle():
	print 'load data'
	we1 = pickle_load("l1_w.pkl")
	we2 = pickle_load("l2_w.pkl")
	we3 = pickle_load("l3_w.pkl")
	we4 = pickle_load("l4_w.pkl")
	hb1 = pickle_load("l1_hb.pkl")
	hb2 = pickle_load("l2_hb.pkl")
	hb3 = pickle_load("l3_hb.pkl")
	hb4 = pickle_load("l4_hb.pkl")
	vb1 = pickle_load("l1_vb.pkl")
	vb2 = pickle_load("l2_vb.pkl")
	vb3 = pickle_load("l3_vb.pkl")
	vb4 = pickle_load("l4_vb.pkl")
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
	train_error = 0.
	for start in xrange(batch): 
		if start % 10 == 0:
			print 'batch', start
		data = insert(converter.train_images[start::batch], converter.dimensionality, 1, axis = 1) 
		w1probs = insert(sigmoid(dot(data, w1)), 1000, 1, axis = 1)
		w2probs = insert(sigmoid(dot(w1probs, w2)), 500, 1, axis = 1)
		w3probs = insert(sigmoid(dot(w2probs, w3)), 250, 1, axis = 1)
		w4probs = insert(dot(w3probs, w4), 30, 1, axis = 1)
		w5probs = insert(sigmoid(dot(w4probs, w5)), 250, 1, axis = 1)
		w6probs = insert(sigmoid(dot(w5probs, w6)), 500, 1, axis = 1)
		w7probs = insert(sigmoid(dot(w6probs, w7)), 1000, 1, axis = 1)
		dataout = sigmoid(dot(w7probs, w8))
		train_error += 1. / len(converter.train_images) * sum((converter.train_images[start::batch] - dataout)  ** 2)
	print train_error / batch

	print '\nstart test error compute'
	test_error = 0.
	for start in xrange(batch): 
		if start % 10 == 0:
			print 'batch', start
		data = insert(converter.test_images[start::batch], converter.dimensionality, 1, axis = 1) 
		w1probs = insert(sigmoid(dot(data, w1)), 1000, 1, axis = 1)
		w2probs = insert(sigmoid(dot(w1probs, w2)), 500, 1, axis = 1)
		w3probs = insert(sigmoid(dot(w2probs, w3)), 250, 1, axis = 1)
		w4probs = insert(dot(w3probs, w4), 30, 1, axis = 1)
		w5probs = insert(sigmoid(dot(w4probs, w5)), 250, 1, axis = 1)
		w6probs = insert(sigmoid(dot(w5probs, w6)), 500, 1, axis = 1)
		w7probs = insert(sigmoid(dot(w6probs, w7)), 1000, 1, axis = 1)
		dataout = sigmoid(dot(w7probs, w8))
		test_error += 1. / len(converter.test_images) * sum((converter.test_images[start::batch] - dataout)  ** 2)
	print test_error / batch


if __name__ == '__main__':
	handle()