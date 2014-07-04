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

if __name__ == '__main__':
    import benchmarks as b
    for func_name, func in  b.__dict__.items():
        if func_name.startswith('benchmark_'):
            func()