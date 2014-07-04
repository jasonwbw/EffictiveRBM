import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import timeit
import time

import worker_inner
from scipy.special import expit

a = np.asarray(np.random.rand(50), dtype=worker_inner.REAL)

A = np.array([[2.0, 0.25, -1.0], 
              [3.0, 0.0 ,  5.0]], dtype=worker_inner.REAL)
B = np.array([[-3.0,  0.5], 
              [ 2.0,  1.5], 
              [ 4.0, -4.0]], dtype=worker_inner.REAL)
a2 = np.asarray(np.random.randn(1, 3), dtype=worker_inner.REAL)
a3 = np.asarray(np.random.randn(1, 2), dtype=worker_inner.REAL)

loops = 1000

def sigmoid_1():
    return worker_inner.sigmoid_2(a)

def sigmoid_2():
    return expit(a)

def benchmark_sigmoid():
    t = timeit.Timer("sigmoid_2()", "from __main__ import sigmoid_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("sigmoid_1()", "from __main__ import sigmoid_1")
    t2 = t.timeit(number=loops)
    print 'sigmoid_1 is %f times faster than sigmoid_2' % (t1 / t2)
    assert (np.linalg.norm(sigmoid_1() - sigmoid_2())) == 0

def dot_1():
    return worker_inner.fast_dot(A, B)

def dot_2():
    return np.dot(A, B)

def benchmark_dot():
    t = timeit.Timer("dot_2()", "from __main__ import dot_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("dot_1()", "from __main__ import dot_1")
    t2 = t.timeit(number=loops)
    print 'dot_1 is %f times faster than dot_2' % (t1 / t2)
    assert np.linalg.norm(dot_1()-dot_2()) == 0

def add_1():
    return worker_inner.add2(A, a2)

def add_2():
    return A + a2

def benchmark_add():
    t = timeit.Timer("add_2()", "from __main__ import add_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("add_1()", "from __main__ import add_1")
    t2 = t.timeit(number=loops)
    print 'add_1 is %f times faster than add_2' % (t1 / t2)
    assert np.linalg.norm(add_1()-add_2()) == 0

def add_random_1():
    return worker_inner.add_random(A, False)

def add_random_2():
    return A + np.random.randn(A.shape[0], A.shape[1]).astype(np.float32)

def benchmark_add_random():
    t = timeit.Timer("add_random_2()", "from __main__ import add_random_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("add_random_1()", "from __main__ import add_random_1")
    t2 = t.timeit(number=loops)
    print 'add_random_1 is %f times faster than add_random_2' % (t1 / t2)

def sum_1():
    return worker_inner.sum(A)

def sum_2():
    return np.sum(A, axis = 0)

def benchmark_sum():
    t = timeit.Timer("sum_2()", "from __main__ import sum_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("sum_1()", "from __main__ import sum_1")
    t2 = t.timeit(number=loops)
    print 'sum_1 is %f times faster than sum_2' % (t1 / t2)
    assert np.linalg.norm(sum_1()-sum_2()) == 0

def square_error_1():
    return worker_inner.square_error(A, A)

def square_error_2():
    return np.sum(A - A)

def benchmark_square_error():
    t = timeit.Timer("square_error_2()", "from __main__ import square_error_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("square_error_1()", "from __main__ import square_error_1")
    t2 = t.timeit(number=loops)
    print 'square_error_1 is %f times faster than square_error_2' % (t1 / t2)
    assert np.linalg.norm(square_error_1()-square_error_2()) == 0

def matrix_multi_1():
    return worker_inner.matrix_multi(A, 3, False)

def matrix_multi_2():
    return A / 3

def benchmark_matrix_multi():
    t = timeit.Timer("matrix_multi_2()", "from __main__ import matrix_multi_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("matrix_multi_1()", "from __main__ import matrix_multi_1")
    t2 = t.timeit(number=loops)
    print 'matrix_multi_1 is %f times faster than matrix_multi_2' % (t1 / t2)
    assert np.linalg.norm(matrix_multi_1()-matrix_multi_2()) == 0

def matrix_plus_1():
    return worker_inner.matrix_plus2(A, A, False)

def matrix_plus_2():
    return A - A

def benchmark_matrix_plus():
    t = timeit.Timer("matrix_plus_2()", "from __main__ import matrix_plus_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("matrix_plus_1()", "from __main__ import matrix_plus_1")
    t2 = t.timeit(number=loops)
    print 'matrix_plus_1 is %f times faster than matrix_plus_2' % (t1 / t2)
    assert np.linalg.norm(matrix_plus_1()-matrix_plus_2()) == 0

def worker_1():
    return worker_inner.fast_worker(A, B, a2, a3, 0.5, 0.5, 0.5, 0.5, False, 1)

from numpy import random, dot, sum, array, exp, zeros, concatenate, float32 as REAL
from scipy.special import expit

def worker(data,\
    weights, hidden_bias, visible_bias,\
    weight_rate, vbias_rate, hbias_rate, weightcost, isLinear, batch_num):
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
        print 'finish batch compute', batch_num, time.asctime(time.localtime(time.time()))
    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

def worker_2():
    return worker(A, B, a3, a2, 0.5, 0.5, 0.5, 0.5, False, 1)

def benchmark_worker():
    t = timeit.Timer("worker_2()", "from __main__ import worker_2")
    t1 = t.timeit(number=loops)
    t = timeit.Timer("worker_1()", "from __main__ import worker_1")
    t2 = t.timeit(number=loops)
    print 'worker_1 is %f times faster than worker_2' % (t1 / t2)

if __name__ == '__main__':
    import benchmarks as b
    for func_name, func in  b.__dict__.items():
        if func_name.startswith('benchmark_'):
            func()