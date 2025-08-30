# Import functions from the compiled Cython module
try:
    # Try relative import first (when used as a package)
    from . import scalapack as _scalapack_cython
except ImportError:
    try:
        # Try direct import if running from within the directory
        import scalapack as _scalapack_cython
    except ImportError:
        raise ImportError(
            "ScaLAPACK Cython module not found. Please build it first using:\n"
            "  pip install -e .\n"
            "in the scalapack_solver directory."
        )

# Import all functions from the Cython module
blacs_pinfo = _scalapack_cython.blacs_pinfo
blacs_get = _scalapack_cython.blacs_get
blacs_gridmap = _scalapack_cython.blacs_gridmap
blacs_gridinfo = _scalapack_cython.blacs_gridinfo
blacs_gridexit = _scalapack_cython.blacs_gridexit
numroc = _scalapack_cython.numroc
indxg2p = _scalapack_cython.indxg2p
descinit = _scalapack_cython.descinit
pdgels = _scalapack_cython.pdgels

import ctypes as ctypes
import numpy as np


def find_work_space(myrow, mycol, nprow, npcol, m, mb_a, n):
    ja, nb_a, nb_b = 1, n, 1
    mb_b = nb_a
    ltau = numroc(ja+n-1, nb_a, mycol, npcol)
    iroffa = 0 % mb_a
    icoffa = 0 % nb_a
    iarow = indxg2p(1, mb_a, myrow, nprow)
    iacol = indxg2p(1, nb_a, mycol, npcol)
    mpa0 = numroc(m+iroffa, mb_a, myrow, nprow, iarow)
    nqa0 = numroc(n+icoffa, nb_a, mycol, npcol, iacol)
    iroffb = 0 % nb_a
    icoffb = 0 % nb_b
    ibrow = indxg2p(1, mb_b, myrow, nprow)
    ibcol = indxg2p(1, nb_b, mycol, npcol)
    npb0 = numroc(n+iroffb, mb_b, myrow, nprow, ibrow)
    nrhsqb0 = numroc(1+icoffb, nb_b, mycol, npcol, ibcol)
    lwf = nb_a * (mpa0 + nqa0 + nb_a)
    lws = max(int((nb_a * (nb_a-1))/2), ((nrhsqb0+npb0)*nb_a)+nb_a*nb_a)
    work_length = ltau + max(lwf, lws)
    return work_length+1


def lstsq(A, b, pt, lengths=None):
    # TODO: calculate size of work matrix needed
    if lengths is None:
        raise ValueError("lengths should not be none!")
    else:
        total_length, node_length, scraped_length = lengths
    
    # The arrays A and b passed here are already the local portions after split_by_node
    # So we work with their actual sizes
    local_rows = len(A)
    
    nprow = pt.get_number_of_nodes()
    npcol = 1
    
    # Calculate block size for ScaLAPACK distribution
    # Each node should have approximately total_length/nprow rows
    mb = int(np.floor(total_length / nprow))
    remainder = total_length - mb * nprow
    
    # First node gets the remainder
    if pt.get_node() == 0:
        mb = mb + remainder
    
    m = total_length  # Global number of rows
    n = np.shape(A)[-1]  # Global number of columns
    nb = n  # Column block size (use full columns)
    
    blacs_pinfo()
    ictxt = blacs_get(0, 0)
    ictxt = blacs_gridmap(ictxt, pt.get_node_list(), nprow, nprow, npcol)
    
    # Get BLACS grid info
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    
    # The local leading dimension should be at least the number of local rows
    # For our case, it's exactly the number of local rows we have
    lld_a = max(1, local_rows)  # Leading dimension must be at least 1
    lld_b = max(1, local_rows)
    
    # Create descriptors with actual local dimensions
    descA = descinit(m, n, mb, nb, ictxt, lld_a)
    descB = descinit(m, 1, mb, 1, ictxt, lld_b)
    
    # Calculate workspace
    work_length = find_work_space(myrow, mycol, nprow, npcol, m, mb, n)
    work = np.zeros(max(1, work_length))  # Ensure work is at least size 1
    
    # Make sure we're using the correct data
    A_local = np.ascontiguousarray(A[:local_rows])  # Ensure contiguous arrays
    b_local = np.ascontiguousarray(b[:local_rows])
    
    # Call ScaLAPACK solver
    pdgels(m, n, 1, A_local, descA, b_local, descB, work, len(work))
    
    # Extract result on rank 0
    if pt.get_rank() == 0:
        b_ptr = b_local.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b_result = np.ctypeslib.as_array(b_ptr, shape=(n,))
    else:
        b_result = None
    
    blacs_gridexit(ictxt)
    pt.all_barrier()
    return b_result


def dummy_lstsq(pt):
    nprow = pt.get_number_of_nodes()
    npcol = 1
    blacs_pinfo()

    ictxt = blacs_get(0, 0)
    ictxt = blacs_gridmap(ictxt, pt.get_node_list(), nprow, nprow, npcol)
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    pt.all_barrier()
    return None
