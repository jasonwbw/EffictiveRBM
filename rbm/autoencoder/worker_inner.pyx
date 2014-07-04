#cython: boundscheck=False
#cython: cdivision=True

cimport cython
import numpy as np
import time

cimport numpy as np

from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX
from scipy.special import expit

REAL = np.float32
ctypedef np.float32_t REAL_t
#end define

cdef REAL_t _inner_sigmoid(REAL_t x):
    return 1. / (1. + exp(-x))

cdef _sigmoid(const unsigned int n_samples, const unsigned int n_features,
              np.ndarray[REAL_t, ndim=2] X,
              np.ndarray[REAL_t, ndim=2] out):
    cdef:
        unsigned int i, j

    for i in range(n_samples):
        for j in range(n_features):
            out[i, j] = _inner_sigmoid(X[i, j])
    return out

cdef np.ndarray[REAL_t, ndim=2] _sigmoid2(np.ndarray[REAL_t, ndim=2] A):
    cdef:
        unsigned int i, j
        np.ndarray[REAL_t, ndim=2] C
        int A_n = A.shape[0]
        int A_m = A.shape[1]

    C = np.zeros((A_n, A_m), dtype = REAL)
    for i in range(A_n):
        for j in range(A_m):
            C[i, j] = _inner_sigmoid(A[i, j])
    return C

def sigmoid_1(X, out=None):
    is_1d = X.ndim == 1
    X = np.asarray(np.atleast_2d(X), dtype = REAL)

    n_samples, n_features = X.shape

    if out is None:
        out = np.empty_like(X)

    _sigmoid(n_samples, n_features, X, out)

    if is_1d:
        return np.squeeze(out)
    return out

def sigmoid_2(v):
    return expit(v)

def sigmoid_3(np.ndarray v):
    cdef np.ndarray e = np.exp(-v)
    return np.clip(1 / (1 + e), 0.00000001, 0.99999999)

fast_sigmoid = sigmoid_2
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

fast_dot = np.dot
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

cdef _add2(REAL_t *C, REAL_t *A, REAL_t *added, const unsigned int n, const unsigned int m):
    cdef unsigned int i, j, index = 0
    for i in range(n):
        for j in range(m):
            C[i * m + j] = A[i * m + j] + added[j]

def add1(A, B):
    return _add(A, B)

def add2(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B):
    cdef np.ndarray[REAL_t, ndim=2] C = np.zeros((A.shape[0], A.shape[1]), dtype=REAL)
    _add2(&C[0, 0], &A[0, 0], &B[0, 0], A.shape[0], A.shape[1])
    return C

fast_add = add2
#end add

cdef np.ndarray[REAL_t, ndim=2] _add_random(np.ndarray[REAL_t, ndim=2] A, const int isLinear):
    cdef:
        int i, j
        float r
        int A_n = A.shape[0]
        int A_m = A.shape[1]
        np.ndarray[REAL_t, ndim=2] C

    C = np.zeros((A_n, A_m), dtype = REAL)
    for i in xrange(A_n):
        for j in xrange(A_m):
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

fast_add_random = add_random
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

fast_sum = sum
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

fast_square_error = square_error
#end square_error

cdef np.ndarray[REAL_t, ndim=2] _matrix_multi(np.ndarray[REAL_t, ndim=2] A, const float p, const int ismulti):
    cdef:
        int i, j
        np.ndarray[REAL_t, ndim=2] C

    C = np.zeros((A.shape[0], A.shape[1]), dtype = REAL)
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            if ismulti == 1:
                C[i, j] = A[i, j] * p
            else:
                C[i, j] = A[i, j] / p
    return C

def matrix_multi(A, p, multi = True):
    if multi:
        return _matrix_multi(A, p, 1)
    else:
        return _matrix_multi(A, p, 0)

fast_matrix_multi = matrix_multi
#end matrix multi

cdef np.ndarray[REAL_t, ndim=2] _matrix_plus(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B, const int isadd):
    cdef:
        int i, j
        np.ndarray[REAL_t, ndim=2] C

    C = np.zeros((A.shape[0], A.shape[1]), dtype = REAL)
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            if isadd == 1:
                C[i, j] = A[i, j] + B[i, j]
            else:
                C[i, j] = A[i, j] - B[i, j]
    return C

cdef _matrix_plus2(REAL_t *C, REAL_t *A, REAL_t *B, const unsigned int n, const unsigned int m, const unsigned int isadd):
    cdef unsigned int i, j
    for i in range(n):
        for j in range(m):
            if isadd == 1:
                C[i * m + j] = A[i * m + j] + B[i * m + j]
            else:
                C[i * m + j] = A[i * m + j] - B[i * m + j]

def matrix_plus(A, B, plus = True):
    if plus:
        return _matrix_plus(A, B, 1)
    else:
        return _matrix_plus(A, B, 0)

def matrix_plus2(np.ndarray[REAL_t, ndim=2] A, np.ndarray[REAL_t, ndim=2] B, plus = True):
    cdef np.ndarray[REAL_t, ndim=2] C = np.zeros((A.shape[0], A.shape[1]), dtype=REAL)
    _matrix_plus2(&C[0, 0], &A[0, 0], &B[0, 0], A.shape[0], A.shape[1], plus and 1 or 0)
    return C

def matrix_plus3(A, B, plus = True):
    if plus:
        return A + B
    else:
        return A - B

fast_matrix_plus = matrix_plus3
#end matrix_plus

cdef _cworker(np.ndarray[REAL_t, ndim=2] data, \
    np.ndarray[REAL_t, ndim=2] weights, np.ndarray[REAL_t, ndim=2] hidden_bias, np.ndarray[REAL_t, ndim=2] visible_bias, \
    const float weight_rate, const float vbias_rate, const float hbias_rate, \
    const float weightcost, const unsigned int isLinear, \
    np.ndarray[REAL_t, ndim=2] add_grad_weight, np.ndarray[REAL_t, ndim=2] add_grad_vbias, np.ndarray[REAL_t, ndim=2] add_grad_hbias, np.ndarray[REAL_t, ndim=2] neg_hidden_probs):
    cdef:
        float error
        np.ndarray[REAL_t, ndim=2] pos_hidden_activations
        np.ndarray[REAL_t, ndim=2] pos_hidden_probs
        np.ndarray[REAL_t, ndim=2] pos_hidden_states
        np.ndarray[REAL_t, ndim=2] posprods
        np.ndarray[REAL_t, ndim=2] neg_visible_probs
        np.ndarray[REAL_t, ndim=2] neg_hidden_activations
        np.ndarray[REAL_t, ndim=2] negprods
    pos_hidden_activations = _add(_dot(data, weights), hidden_bias)
    pos_hidden_probs = (isLinear == 1 and pos_hidden_activations or sigmoid_2(pos_hidden_activations))
    pos_hidden_states = _add_random(pos_hidden_probs, isLinear)
    posprods = _dot(data.T, pos_hidden_probs)

    neg_visible_probs = sigmoid_2(_add(fast_dot(pos_hidden_states, weights.T), visible_bias))
    neg_hidden_activations = _add(fast_dot(neg_visible_probs, weights), hidden_bias)
    neg_hidden_probs = (isLinear == 1 and neg_hidden_activations or sigmoid_2(neg_hidden_activations))
    negprods = _dot(neg_visible_probs.T, neg_hidden_probs)

    add_grad_weight = weight_rate * ((posprods - negprods) / len(data) - weightcost * weights)
    add_grad_vbias = vbias_rate * (_sum(data) - _sum(neg_visible_probs)) / len(data)
    add_grad_hbias = hbias_rate * (_sum(pos_hidden_probs) - _sum(neg_hidden_probs)) / len(data)

    error = _square_error(data, neg_visible_probs)

    return error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs

def cworker2(data, \
    weights, hidden_bias, visible_bias, \
    weight_rate, vbias_rate, hbias_rate, weightcost, \
    isLinear, batch_num):
    add_grad_weight = np.zeros(weights.shape, dtype = REAL)
    add_grad_vbias = np.zeros(visible_bias.shape, dtype = REAL)
    add_grad_hbias = np.zeros(hidden_bias.shape, dtype = REAL)
    neg_hidden_probs = np.zeros((data.shape[0], weights.shape[1]), dtype = REAL)
    error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs = _cworker(data, weights, hidden_bias, visible_bias,
        weight_rate, vbias_rate, hbias_rate, weightcost, 
        isLinear, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)
    if batch_num % 10 == 0:
        print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )
    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

def cworker1(data, \
    weights, hidden_bias, visible_bias, \
    weight_rate, vbias_rate, hbias_rate, weightcost, \
    isLinear, batch_num):
    pos_hidden_activations = fast_add(fast_dot(data, weights), hidden_bias)
    pos_hidden_probs = isLinear and pos_hidden_activations or fast_sigmoid(pos_hidden_activations)
    pos_hidden_states = fast_add_random(pos_hidden_probs, isLinear)
    posprods = fast_dot(data.T, pos_hidden_probs)
    pos_hidden_act = fast_sum(pos_hidden_probs)
    pos_visible_act = fast_sum(data)

    neg_visible_activations = fast_add(fast_dot(pos_hidden_states, weights.T), visible_bias)
    neg_visible_probs = fast_sigmoid(neg_visible_activations)
    neg_hidden_activations = fast_add(fast_dot(neg_visible_probs, weights), hidden_bias)
    if isLinear:
        neg_hidden_probs = neg_hidden_activations
    else:
        neg_hidden_probs = fast_sigmoid(neg_hidden_activations)
    negprods = fast_dot(neg_visible_probs.T, neg_hidden_probs)
    neg_hidden_act = fast_sum(neg_hidden_probs)
    neg_visible_act = fast_sum(neg_visible_probs)

    add_grad_weight = fast_matrix_multi(fast_matrix_multi(fast_matrix_plus(posprods, negprods, False), len(data), False) - fast_matrix_multi(weights, weightcost), weight_rate)
    add_grad_vbias = fast_matrix_multi(fast_matrix_plus(pos_visible_act, neg_visible_act, False), vbias_rate / len(data))
    add_grad_hbias = fast_matrix_multi(fast_matrix_plus(pos_hidden_act, neg_hidden_act, False), hbias_rate / len(data))

    error = fast_square_error(data, neg_visible_probs)

    if batch_num % 10 == 0:
        print 'finish batch compute', batch_num, time.asctime( time.localtime(time.time()) )

    return (error, add_grad_weight, add_grad_vbias, add_grad_hbias, neg_hidden_probs)

fast_worker = cworker1