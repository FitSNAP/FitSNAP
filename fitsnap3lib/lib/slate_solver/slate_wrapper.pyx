# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
from mpi4py import MPI
from mpi4py.libmpi cimport MPI_Comm

np.import_array()

cdef extern from "slate/slate.hh" namespace "slate":
    void initialize() except +
    void finalize() except +
    
cdef extern from "slate_ridge.cpp":
    void slate_ridge_solve(double* local_ata, double* local_atb, double* solution,
                          int n, double alpha, MPI_Comm comm, int tile_size) except +

def ridge_solve(np.ndarray[double, ndim=2, mode="c"] local_ata,
                np.ndarray[double, ndim=1, mode="c"] local_atb,
                double alpha,
                MPI.Comm comm,
                int tile_size=256):
    """
    Solve ridge regression using SLATE.
    
    Args:
        local_ata: Local A^T A matrix
        local_atb: Local A^T b vector
        alpha: Ridge parameter
        comm: MPI communicator
        tile_size: SLATE tile size
    
    Returns:
        Solution vector
    """
    cdef int n = local_ata.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] solution = np.zeros(n, dtype=np.float64)
    cdef MPI_Comm c_comm = comm.ob_mpi
    
    slate_ridge_solve(&local_ata[0,0], &local_atb[0], &solution[0], 
                     n, alpha, c_comm, tile_size)
    
    return solution
