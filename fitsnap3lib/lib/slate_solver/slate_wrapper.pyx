# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stddef cimport size_t

cdef extern from *:
    """
    extern "C" void slate_augmented_qr(double* local_a, double* local_b, int m, int n, int lld, int debug);
    """
    void slate_augmented_qr(double* local_a, double* local_b, int m, int n, int lld, int debug) except +

def slate_augmented_qr_cython(double[::1, :] local_a, double[::1] local_b, int m, int lld, int debug=0):

    cdef int n = <int>local_a.shape[1]
    slate_augmented_qr(&local_a[0, 0], &local_b[0], m, n, lld, debug)
