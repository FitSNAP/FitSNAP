# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stddef cimport size_t

cdef extern from *:
    """
    extern "C" void slate_ridge_solve_qr(double* local_a, double* local_b,
                                          int m, int n, int lld, void* comm, int tile_size);
    """
    void slate_ridge_solve_qr(double* local_a, double* local_b,
                              int m, int n, int lld, void* comm, int tile_size) except +

ctypedef void* MPI_Comm

def ridge_solve_qr(double[::1, :] local_a,  # F-contiguous (column-major) for SLATE
                   double[::1]     local_b,
                   int             m,
                   int             lld,
                   comm,
                   int             tile_size=32*1024*1024): # 32MB
                   
    cdef int n = <int>local_a.shape[1]
    from mpi4py import MPI
    cdef size_t comm_ptr = MPI._handleof(comm)
    slate_ridge_solve_qr(&local_a[0, 0], &local_b[0], m, n, lld, <void*>comm_ptr, tile_size)
