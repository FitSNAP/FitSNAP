# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stddef cimport size_t

cdef extern from *:
    """
    extern "C" void slate_ridge_solve_qr(double* local_a, double* local_b, double* solution,
                                          int m_local, int m, int n, double alpha, void* comm, int tile_size);
    """
    void slate_ridge_solve_qr(double* local_a, double* local_b, double* solution,
                              int m_local, int m, int n, double alpha, void* comm, int tile_size) except +

ctypedef void* MPI_Comm

def ridge_solve_qr(double[:, ::1] local_a,
                   double[::1]     local_b,
                   int             m,
                   double          alpha,
                   comm,
                   int             tile_size=256):
    cdef int m_local = <int>local_a.shape[0]
    cdef int n       = <int>local_a.shape[1]

    if local_b.shape[0] != m_local:
        raise ValueError("local_b length must equal local_a.shape[0].")

    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

    from mpi4py import MPI
    cdef size_t comm_ptr = MPI._handleof(comm)

    cdef double* a_ptr = NULL
    cdef double* b_ptr = NULL
    cdef double* x_ptr = NULL

    if m_local > 0:
        a_ptr = &local_a[0, 0]
        b_ptr = &local_b[0]
    if n > 0:
        x_ptr = <double*>&x[0]

    slate_ridge_solve_qr(a_ptr, b_ptr, x_ptr, m_local, m, n, alpha, <void*>comm_ptr, tile_size)
    return x
