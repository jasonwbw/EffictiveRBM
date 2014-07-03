#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author : Jasonwbw@yahoo.com

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
	reload_support=True)

from worker_inner import fast_sigmoid as sigmoid
from worker_inner import fast_dot as dot
from worker_inner import add, add_random, sum, square_error

from numpy import random
from scipy.special import expit

if __name__ == '__main__':
	a = random.randn(3, 5).astype(np.float32)
	print sigmoid(a)
	print expit(a)
	print ''

	b = random.randn(5, 2).astype(np.float32)
	print dot(a, b)
	print np.dot(a, b)
	print ''

	c = random.randn(1, 2).astype(np.float32)
	print add(b, c)
	print b + c
	print ''

	print sigmoid(a)
	print add_random(sigmoid(a), True)
	print add_random(sigmoid(a), False)
	print ''

	print a
	print sum(a)
	print np.sum(a, axis = 0)
	print ''

	d = random.randn(5, 2).astype(np.float32)
	print b
	print d
	print square_error(d, b)