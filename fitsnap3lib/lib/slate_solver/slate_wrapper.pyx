# cython: language_level=3
# distutils: language = c++
# distutils: define_macros = NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

np.import_array()

# Define MPI_Comm as a void pointer (opaque type)
ctypedef void* MPI_Comm

cdef extern from "slate/slate.hh" namespace "slate":
    void initialize() except +
    void finalize() except +

# Declare the C++ functions
cdef extern from *:
    """
    extern "C" void slate_ridge_solve_qr(double* local_a, double* local_b, double* solution,
                                          int m_local, int n, double alpha, void* comm, int tile_size);
    """
    void slate_ridge_solve_qr(double* local_a, double* local_b, double* solution,
                              int m_local, int n, double alpha, void* comm, int tile_size) except +

def ridge_solve_qr(np.ndarray[double, ndim=2, mode="c"] local_a,
                    np.ndarray[double, ndim=1, mode="c"] local_b,
                    double alpha,
                    comm,
                    int tile_size=256):
    """
    Solve ridge regression using SLATE with augmented least squares and QR.
    
    Args:
        local_a: Local portion of matrix A (m_local x n)
        local_b: Local portion of vector b (m_local,)
        alpha: Ridge parameter
        comm: MPI communicator (mpi4py.MPI.Comm object)
        tile_size: SLATE tile size
    
    Returns:
        Solution vector (n,)
    """
    cdef int m_local = local_a.shape[0]
    cdef int n = local_a.shape[1]
    cdef np.ndarray[double, ndim=1, mode="c"] solution = np.zeros(n, dtype=np.float64)
    
    # Get the MPI communicator handle at runtime
    from mpi4py import MPI
    cdef size_t comm_ptr = MPI._handleof(comm)
    
    # Handle empty arrays (m_local = 0)
    cdef double* a_ptr = NULL
    cdef double* b_ptr = NULL
    
    if m_local > 0:
        a_ptr = &local_a[0,0]
        b_ptr = &local_b[0]
    
    slate_ridge_solve_qr(a_ptr, b_ptr, &solution[0], 
                        m_local, n, alpha, <void*>comm_ptr, tile_size)
    
    return solution
