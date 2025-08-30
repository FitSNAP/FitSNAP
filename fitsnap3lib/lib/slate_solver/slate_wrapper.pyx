# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np

np.import_array()

# Define MPI_Comm as a void pointer (opaque type)
ctypedef void* MPI_Comm

cdef extern from "slate/slate.hh" namespace "slate":
    void initialize() except +
    void finalize() except +

# Declare the C++ function (it's defined in slate_ridge.cpp)
cdef extern from *:
    """
    extern "C" void slate_ridge_solve(double* local_ata, double* local_atb, double* solution,
                                      int n, double alpha, void* comm, int tile_size);
    """
    void slate_ridge_solve(double* local_ata, double* local_atb, double* solution,
                          int n, double alpha, void* comm, int tile_size) except +

def ridge_solve(np.ndarray[double, ndim=2, mode="c"] local_ata,
                np.ndarray[double, ndim=1, mode="c"] local_atb,
                double alpha,
                comm,
                int tile_size=256):
    """
    Solve ridge regression using SLATE.
    
    Args:
        local_ata: Local A^T A matrix
        local_atb: Local A^T b vector
        alpha: Ridge parameter
        comm: MPI communicator (mpi4py.MPI.Comm object)
        tile_size: SLATE tile size
    
    Returns:
        Solution vector
    """
    cdef int n = local_ata.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] solution = np.zeros(n, dtype=np.float64)
    
    # Get the MPI communicator handle at runtime
    # Import mpi4py at runtime to avoid build-time dependency
    from mpi4py import MPI
    cdef size_t comm_ptr = MPI._handleof(comm)
    
    slate_ridge_solve(&local_ata[0,0], &local_atb[0], &solution[0], 
                     n, alpha, <void*>comm_ptr, tile_size)
    
    return solution
