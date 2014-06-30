#cython: boundscheck=False
#cython: cdivision=True

cimport cython
import numpy as np
import time

cimport numpy as np

from libc.math cimport exp

REAL = np.float32
ctypedef np.float32_t REAL_t

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

cdef np.ndarray[REAL_t, ndim=2] _dot(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B):
    cdef: 
        int i, j, k
        int A_n = A.shape[0]
        int A_m = A.shape[1]
        int B_n = B.shape[0]
        int B_m = B.shape[1]
        np.ndarray[double, ndim=2] C
    
    # Are matrices conformable?
    assert A_m == B_n, \
        'Non-conformable shapes.'
    
    # Initialize the results matrix.
    C = np.zeros((A_n, B_m))
    for i in xrange(A_n):
        for j in xrange(B_m):
            for k in xrange(A_m):
                C[i, j] += A[i, k] * B[k, j]
    return C

def dot(A, B):
    A = np.asarray(np.atleast_2d(A), dtype = REAL)
    B = np.asarray(np.atleast_2d(B), dtype = REAL)
    return _dot(A, B)

def cworker(data, 
    weights, hidden_bias, visible_bias, 
    weight_rate, vbias_rate, hbias_rate, weightcost, 
    isLinear, batch_num):
    pos_hidden_activations = dot(data, weights) + hidden_bias
    if isLinear:
        pos_hidden_probs = pos_hidden_activations
        pos_hidden_states = pos_hidden_probs + np.random.randn(len(data), len(hidden_bias)).astype(REAL)
    else:
        pos_hidden_probs = sigmoid(pos_hidden_activations)
        pos_hidden_states = pos_hidden_probs > np.random.randn(len(data), len(hidden_bias)).astype(REAL)
    posprods = dot(data.T, pos_hidden_probs)
    pos_hidden_act = np.sum(pos_hidden_probs)
    pos_visible_act = np.sum(data)

    neg_visible_activations = dot(pos_hidden_states, weights.T) + visible_bias
    neg_visible_probs = sigmoid(neg_visible_activations)
    neg_hidden_activations = dot(neg_visible_probs, weights) + hidden_bias
    if isLinear:
        neg_hidden_probs = neg_hidden_activations
    else:
        neg_hidden_probs = sigmoid(neg_hidden_activations)
    negprods = dot(neg_visible_probs.T, neg_hidden_probs)
    neg_hidden_act = np.sum(neg_hidden_probs)
    neg_visible_act = np.sum(neg_visible_probs)

    add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
    add_grad_vbias = vbias_rate * (pos_visible_act - neg_visible_act) / len(data)
    add_grad_hbias = hbias_rate * (pos_hidden_act - neg_hidden_act) / len(data)

    error = np.sum((data - neg_visible_probs) ** 2)

    if batch_num % 10 == 0:
        print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )

    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)