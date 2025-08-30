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
blacs_gridinit = _scalapack_cython.blacs_gridinit
blacs_gridmap = _scalapack_cython.blacs_gridmap
blacs_gridinfo = _scalapack_cython.blacs_gridinfo
blacs_gridexit = _scalapack_cython.blacs_gridexit
numroc = _scalapack_cython.numroc
indxg2p = _scalapack_cython.indxg2p
descinit = _scalapack_cython.descinit
pdgels = _scalapack_cython.pdgels

import ctypes as ctypes
import numpy as np


def find_work_space(m, n, mb, nb, myrow, mycol, nprow, npcol):
    """Calculate workspace needed for PDGELS.
    
    Based on ScaLAPACK documentation for PDGELS workspace calculation.
    """
    # Calculate local dimensions
    local_m = numroc(m, mb, myrow, nprow, 0)
    local_n = numroc(n, nb, mycol, npcol, 0)
    
    # For PDGELS, the workspace formula from the documentation:
    # If M >= N: LWORK >= NB * (local_m + local_n + NB)
    # If M < N: LWORK >= MB * (local_m + local_n + MB)
    
    if m >= n:
        # Overdetermined or square system
        work_length = nb * (local_m + local_n + nb)
    else:
        # Underdetermined system  
        work_length = mb * (local_m + local_n + mb)
    
    # Add some safety margin
    work_length = int(work_length * 1.1) + 1000
    
    # Sanity check to prevent massive memory allocations
    max_reasonable_work = 10000000  # 10 million doubles ~80MB
    if work_length > max_reasonable_work:
        print(f"WARNING: Calculated work_length={work_length} is too large, capping at {max_reasonable_work}")
        work_length = max_reasonable_work
    
    return max(1, work_length)


def lstsq(A, b, pt, lengths=None):
    """Solve least squares problem using ScaLAPACK PDGELS.
    
    This function should be called by the root process of each node.
    It distributes the matrix across nodes using ScaLAPACK's 2D block-cyclic distribution.
    """
    if lengths is None:
        raise ValueError("lengths should not be none!")
    else:
        total_length, node_length, scraped_length = lengths
    
    # Debug print the inputs
    if pt.get_subrank() == 0:
        print(f"[Node {pt.get_node()}] lstsq called with:", flush=True)
        print(f"  lengths = {lengths}", flush=True)
        print(f"  A.shape = {A.shape}", flush=True)
        print(f"  b.shape = {b.shape}", flush=True)
    
    # The arrays A and b passed here are already the local portions after split_by_node
    local_rows = len(A)
    
    # Additional validation
    if local_rows == 0:
        raise ValueError(f"[Node {pt.get_node()}] No data on this node! A.shape={A.shape}")
    
    if total_length <= 0 or total_length > 1e9:  # Sanity check on global size
        raise ValueError(f"[Node {pt.get_node()}] Invalid total_length={total_length}")
    
    # Debug actual array content
    if pt.get_subrank() == 0:
        print(f"[Node {pt.get_node()}] Local data:", flush=True)
        print(f"  local_rows = {local_rows}", flush=True)
        print(f"  First row of A: {A[0][:5] if len(A) > 0 and len(A[0]) > 0 else 'empty'}...", flush=True)
        print(f"  First element of b: {b[0] if len(b) > 0 else 'empty'}", flush=True)
    
    # Set up process grid dimensions
    # For multi-node runs, we use one process per node
    nprow = pt.get_number_of_nodes()
    npcol = 1
    
    # Use reasonable block sizes for ScaLAPACK
    # Block size should be small for cache efficiency, not the total rows per node
    mb = min(64, total_length)  # Row block size (typically 32-128)
    nb = min(64, n := np.shape(A)[-1])  # Column block size
    
    m = total_length  # Global number of rows
    n = np.shape(A)[-1]  # Global number of columns
    
    # Initialize BLACS
    rank, nprocs = blacs_pinfo()
    
    if pt.get_subrank() == 0:
        print(f"[Node {pt.get_node()}] BLACS pinfo: rank={rank}, nprocs={nprocs}", flush=True)
    
    # Get BLACS context
    ictxt = blacs_get(0, 0)
    
    # For ScaLAPACK, we need to create a process grid
    # When using multiple nodes, each node's root process participates
    
    if pt.get_subrank() == 0:
        print(f"[Node {pt.get_node()}] Setting up BLACS grid:", flush=True)
        print(f"  nprow={nprow}, npcol={npcol}", flush=True)
        print(f"  MPI rank={pt.get_rank()}", flush=True)
    
    # Initialize the grid with row-major layout
    # This assigns processes to the grid in row-major order
    ictxt = blacs_gridinit(ictxt, 'R', nprow, npcol)
    
    # Get BLACS grid info
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    
    # Calculate the number of local rows using numroc
    # This tells us how many rows this process should have
    # numroc(n, nb, iproc, nprocs, srcproc)
    local_m = numroc(m, mb, myrow, nprow, 0)  # srcproc=0
    local_n = numroc(n, nb, mycol, npcol, 0)  # srcproc=0
    
    # The local leading dimension should be at least the number of local rows
    # as calculated by numroc (not the actual array size we have)
    lld_a = max(1, local_m)  # Leading dimension must be at least 1
    lld_b = max(1, local_m)
    
    # Debug output
    if pt.get_subrank() == 0:
        print(f"[Node {pt.get_node()}] ScaLAPACK grid info:", flush=True)
        print(f"  myrow={myrow}, mycol={mycol}", flush=True)
        print(f"  mb={mb}, nb={nb}", flush=True)
        print(f"  local_m={local_m}, local_n={local_n}", flush=True)
        print(f"  lld_a={lld_a}, lld_b={lld_b}", flush=True)
    
    # Create descriptors with calculated local dimensions
    descA = descinit(m, n, mb, nb, ictxt, lld_a)
    descB = descinit(m, 1, mb, 1, ictxt, lld_b)
    
    # Calculate workspace
    work_length = find_work_space(m, n, mb, nb, myrow, mycol, nprow, npcol)
    work = np.zeros(max(1, work_length), dtype=np.float64)  # Ensure work is at least size 1
    
    # Prepare local arrays that match ScaLAPACK's expected dimensions
    # ScaLAPACK expects local_m rows and local_n columns for A
    # and local_m rows for b
    if local_rows != local_m:
        if pt.get_subrank() == 0:
            print(f"[Node {pt.get_node()}] WARNING: local_rows ({local_rows}) != local_m ({local_m})", flush=True)
            print(f"  Adjusting array sizes for ScaLAPACK...", flush=True)
    
    # Create properly sized arrays for ScaLAPACK
    A_local = np.zeros((local_m, local_n), dtype=np.float64, order='F')  # Fortran order
    b_local = np.zeros((local_m, 1), dtype=np.float64, order='F')  # b is 2D for ScaLAPACK
    
    # Copy available data into ScaLAPACK arrays
    min_rows = min(local_rows, local_m)
    min_cols = min(n, local_n)
    A_local[:min_rows, :min_cols] = A[:min_rows, :min_cols]
    b_local[:min_rows, 0] = b[:min_rows]
    
    # Call ScaLAPACK solver
    pdgels(m, n, 1, A_local, descA, b_local, descB, work, len(work))
    
    # Extract result on rank 0
    # The solution is stored in the first n elements of b_local on process 0
    if pt.get_rank() == 0:
        # Extract the solution vector (first n elements of b_local)
        b_result = b_local[:n, 0].copy()  # Get first n elements from the column
    else:
        b_result = None
    
    blacs_gridexit(ictxt)
    pt.all_barrier()
    return b_result


def dummy_lstsq(pt):
    """Dummy lstsq function for non-root processes in ScaLAPACK.
    
    This ensures all processes participate in the BLACS grid even if they don't have data.
    Only processes with subrank != 0 should call this.
    """
    nprow = pt.get_number_of_nodes()
    npcol = 1
    
    # Initialize BLACS
    rank, nprocs = blacs_pinfo()
    
    # Get BLACS context
    ictxt = blacs_get(0, 0)
    
    # Initialize the grid with row-major layout (must match lstsq)
    ictxt = blacs_gridinit(ictxt, 'R', nprow, npcol)
    
    # Get grid info (all processes must call this)
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    
    # Exit the grid
    blacs_gridexit(ictxt)
    
    # Synchronize
    pt.all_barrier()
    return None
