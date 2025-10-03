# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t
from libc.stddef cimport size_t


cdef extern from *:
    """
    #include <cstdint>
    
    extern "C" {
        void slate_ridge_augmented_qr(double* local_aw, double* local_bw,
                                      int64_t m, int64_t n, int64_t lld, int debug);
        
        void slate_ard_update_sigma(double* local_aw, double* local_sigma,
                                    int64_t m, int64_t n, int64_t lld,
                                    double alpha, double* lambda_arr, unsigned char* keep_lambda, 
                                    int64_t n_active, int debug);
        
        void slate_ard_update_coeff(double* local_aw, double* local_bw, double* local_coef,
                                    int64_t m, int64_t n, int64_t lld, double alpha,
                                    unsigned char* keep_lambda, int64_t n_active, 
                                    double* sigma, int debug);
    }
    """

    void slate_ridge_augmented_qr(double* local_aw, double* local_bw, int64_t m, int64_t n, int64_t lld, int debug) except +
    
    void slate_ard_update_sigma(double* local_aw, double* local_sigma, int64_t m, int64_t n, int64_t lld,
                                double alpha, double* lambda_arr, np.npy_bool* keep_lambda,
                                int64_t n_active, int debug) except +
    
    void slate_ard_update_coeff(double* local_aw, double* local_bw, double* local_coef,
                                int64_t m, int64_t n, int64_t lld, double alpha,
                                np.npy_bool* keep_lambda, int64_t n_active, 
                                double* sigma, int debug) except +

def slate_ridge_augmented_qr_cython(double[::1, :] local_aw, double[::1] local_bw, int m, int lld, int debug=0):
    cdef int n = <int>local_aw.shape[1]
    slate_ridge_augmented_qr(&local_aw[0, 0], &local_bw[0], m, n, lld, debug)

def slate_ard_update_cython(double[::1, :] local_aw, double[::1] local_bw,
                           double[::1] coef, double alpha, double[::1] lambda_arr,
                           np.ndarray[np.npy_bool, ndim=1, cast=True] keep_lambda,
                           int m, int debug=0):
    """
    Perform one iteration of ARD updates: compute sigma and coefficients.
    
    Parameters:
    -----------
    local_a : (lld, n) design matrix (local portion, column-major)
    local_b : (lld,) target vector (local portion)
    local_w : (lld,) weights (not used in ARD, but kept for API consistency)
    coef : (n,) current coefficients
    alpha : float, noise precision
    lambda_arr : (n,) feature precisions
    keep_lambda : (n,) bool mask of active features
    m : int, global number of samples
    debug : int, debug flag
    
    Returns:
    --------
    sigma : (n_active, n_active) posterior covariance matrix
    coef_new : (n,) updated coefficients
    """
    cdef int n = <int>local_aw.shape[1]
    cdef int lld = <int>local_aw.shape[0]
    cdef int64_t n_active = np.sum(keep_lambda)
    
    # Allocate output arrays - use simple np.ndarray without typed memoryview
    cdef np.ndarray sigma = np.zeros((n_active, n_active), dtype=np.float64, order='F')
    cdef np.ndarray coef_new = np.asarray(coef).copy()  # Convert memoryview to ndarray first
    cdef double* sigma_ptr
    cdef double* coef_new_ptr
    
    if n_active == 0:
        return sigma, coef_new
    
    # Get data pointers
    sigma_ptr = <double*>np.PyArray_DATA(sigma)
    coef_new_ptr = <double*>np.PyArray_DATA(coef_new)
    
    # Update sigma: inv(alpha * X.T @ X + diag(lambda))
    slate_ard_update_sigma(&local_aw[0, 0], sigma_ptr, m, n, lld, alpha,
                          &lambda_arr[0], <np.npy_bool*>&keep_lambda[0], n_active, debug)
    
    # Update coefficients: alpha * sigma @ X.T @ y
    slate_ard_update_coeff(&local_aw[0, 0], &local_bw[0], coef_new_ptr,
                          m, n, lld, alpha, <np.npy_bool*>&keep_lambda[0],
                          n_active, sigma_ptr, debug)
    
    return sigma, coef_new
