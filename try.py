from numba import jit
from numpy import arange
import numpy as np
import time


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
#@jit

@jit
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    start_time = time.time()
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    print("--- %s seconds ---" % (time.time() - start_time))
    return D


def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    start_time = time.time()
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    print("--- %s seconds ---" % (time.time() - start_time))
    return result


def sum1d(arr):
    M, N = arr.shape
    result = 0.0
    start_time = time.time()
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    print("--- %s seconds ---" % (time.time() - start_time))
    return result


X = np.random.random((1000, 3))
ww = pairwise_python(X)
#pairwise_numba = autojit(pairwise_python)
#pairwise_numba(X)

