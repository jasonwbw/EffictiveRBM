#cython: boundscheck=False
#cython: cdivision=True

cimport cython
import numpy as np
import time

cimport numpy as np

from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX

REAL = np.float32
ctypedef np.float32_t REAL_t
#end define

cdef REAL_t _inner_sigmoid(REAL_t x):
    return 1. / (1. + exp(-x))

cdef _sigmoid(const unsigned int n_samples, const unsigned int n_features,
              np.ndarray[REAL_t, ndim=2] X,
              np.ndarray[REAL_t, ndim=2] out):
    cdef unsigned int i, j
    for i in range(n_samples):
        for j in range(n_features):
            out[i, j] = _inner_sigmoid(X[i, j])
    return out

def sigmoid(X, out=None):
    is_1d = X.ndim == 1
    X = np.asarray(np.atleast_2d(X), dtype = REAL)

    n_samples, n_features = X.shape

    if out is None:
        out = np.empty_like(X)

    _sigmoid(n_samples, n_features, X, out)

    if is_1d:
        return np.squeeze(out)
    return out

fast_sigmoid = sigmoid
#end sigmoid

cdef np.ndarray[REAL_t, ndim=2] _dot(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B):
    cdef: 
        int i, j, k
        int A_n = A.shape[0]
        int A_m = A.shape[1]
        int B_n = B.shape[0]
        int B_m = B.shape[1]
        np.ndarray[REAL_t, ndim=2] C
    
    # Are matrices conformable?
    assert A_m == B_n, \
        'Non-conformable shapes.'
    
    # Initialize the results matrix.
    C = np.zeros((A_n, B_m), dtype = REAL)
    for i in xrange(A_n):
        for j in xrange(B_m):
            for k in xrange(A_m):
                C[i, j] += A[i, k] * B[k, j]
    return C

def dot(A, B):
    A = np.asarray(np.atleast_2d(A), dtype = REAL)
    B = np.asarray(np.atleast_2d(B), dtype = REAL)
    return _dot(A, B)

fast_dot = dot
#end dot

cdef np.ndarray[REAL_t, ndim=2] _add(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] added):
    cdef:
        int i, j
        int A_n = A.shape[0]
        int A_m = A.shape[1]
        int added_m = added.shape[1]
        np.ndarray[REAL_t, ndim=2] C

    assert A_m == added_m, \
        'Non-conformable shapes.'

    C = np.zeros((A_n, A_m), dtype = REAL)
    for i in xrange(A_n):
        for j in xrange(A_m):
            C[i, j] = A[i, j] + added[0, j]
    return C

def add(A, B):
    return _add(A, B)
#end add

cdef np.ndarray[REAL_t, ndim=2] _add_random(np.ndarray[REAL_t, ndim=2] A, const int isLinear):
    cdef:
        int i, j
        float r
        int A_n = A.shape[0]
        int A_m = A.shape[1]
        np.ndarray[REAL_t, ndim=2] C

    C = np.zeros((A_n, A_m), dtype = REAL)
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            if isLinear == 1:
                C[i, j] = A[i, j] + rand() / float(RAND_MAX)
            else:
                r = rand() / float(RAND_MAX)
                if A[i, j] > r:
                    C[i, j] = 1
                else:
                    C[i, j] = 0
    return C 

def add_random(A, isLinear):
    if isLinear:
        return _add_random(A, 1)
    else:
        return _add_random(A, 0)
#end add_random

cdef np.ndarray[REAL_t, ndim=2] _sum(np.ndarray[REAL_t, ndim=2] A):
    cdef:
        int i, j
        np.ndarray[REAL_t, ndim=2] C

    C = np.zeros((1, A.shape[1]), dtype = REAL)
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            C[0, j] += A[i, j]
    return C

def sum(A):
    return _sum(A)
#end sum

cdef float _square_error(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B):
    cdef:
        int i, j
        float s = 0.0

    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            s += (A[i, j] - B[i, j]) * (A[i, j] - B[i, j])
    return s

def square_error(A, B):
    return _square_error(A, B)
#end square_error

def cworker2(data, \
    weights, hidden_bias, visible_bias, \
    weight_rate, vbias_rate, hbias_rate, weightcost, \
    isLinear, batch_num):
    pos_hidden_activations = add(fast_dot(data, weights), hidden_bias)
    pos_hidden_probs = isLinear and pos_hidden_activations or fast_sigmoid(pos_hidden_activations)
    pos_hidden_states = add_random(pos_hidden_probs, isLinear)
    posprods = fast_dot(data.T, pos_hidden_probs)
    pos_hidden_act = sum(pos_hidden_probs)
    pos_visible_act = sum(data)

    neg_visible_activations = add(fast_dot(pos_hidden_states, weights.T), visible_bias)
    neg_visible_probs = fast_sigmoid(neg_visible_activations)
    neg_hidden_activations = add(fast_dot(neg_visible_probs, weights), hidden_bias)
    if isLinear:
        neg_hidden_probs = neg_hidden_activations
    else:
        neg_hidden_probs = fast_sigmoid(neg_hidden_activations)
    negprods = fast_dot(neg_visible_probs.T, neg_hidden_probs)
    neg_hidden_act = sum(neg_hidden_probs)
    neg_visible_act = sum(neg_visible_probs)

    add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
    add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
    add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)

    error = square_error(data, neg_visible_probs)

    if batch_num % 10 == 0:
        print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )

    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

def cworker1(data, \
    weights, hidden_bias, visible_bias, \
    weight_rate, vbias_rate, hbias_rate, weightcost, \
    isLinear, batch_num):
    pos_hidden_activations = fast_dot(data, weights) + hidden_bias
    if isLinear:
        pos_hidden_probs = pos_hidden_activations
        pos_hidden_states = pos_hidden_probs + np.random.randn(len(data), len(hidden_bias)).astype(REAL)
    else:
        pos_hidden_probs = fast_sigmoid(pos_hidden_activations)
        pos_hidden_states = pos_hidden_probs > np.random.randn(len(data), len(hidden_bias)).astype(REAL)
    posprods = fast_dot(data.T, pos_hidden_probs)
    pos_hidden_act = np.sum(pos_hidden_probs, axis = 0)
    pos_visible_act = np.sum(data, axis = 0)

    neg_visible_activations = fast_dot(pos_hidden_states, weights.T) + visible_bias
    neg_visible_probs = fast_sigmoid(neg_visible_activations)
    neg_hidden_activations = fast_dot(neg_visible_probs, weights) + hidden_bias
    if isLinear:
        neg_hidden_probs = neg_hidden_activations
    else:
        neg_hidden_probs = fast_sigmoid(neg_hidden_activations)
    negprods = fast_dot(neg_visible_probs.T, neg_hidden_probs)
    neg_hidden_act = np.sum(neg_hidden_probs, axis = 0)
    neg_visible_act = np.sum(neg_visible_probs, axis = 0)

    add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
    add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
    add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)

    error = np.sum((data - neg_visible_probs) ** 2)

    if batch_num % 10 == 0:
        print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )

    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

fast_worker = cworker2