# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t
from libc.stddef cimport size_t


cdef extern from *:

    void slate_ridge_augmented_qr(double* local_a, double* local_b, int64_t m, int64_t n, int64_t lld, int debug) except +
    void slate_ard(double* local_a, double* local_b, double* local_diag, int64_t m, int64_t n, int64_t lld, int debug) except +

def slate_ridge_augmented_qr_cython(double[::1, :] local_a, double[::1] local_b, int m, int lld, int debug=0):
    cdef int n = <int>local_a.shape[1]
    slate_ridge_augmented_qr(&local_a[0, 0], &local_b[0], m, n, lld, debug)

def slate_augmented_qr_with_diag_cython(double[::1, :] local_a, double[::1] local_b, double[::1] local_diag, int m, int lld, int debug=0):
    cdef int n = <int>local_a.shape[1]
    slate_augmented_qr_with_diag(&local_a[0, 0], &local_b[0], &local_diag[0], m, n, lld, debug)
