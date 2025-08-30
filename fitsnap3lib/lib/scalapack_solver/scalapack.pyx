# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from Scalapack cimport *
from libc.stdlib cimport calloc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np

# Initialize numpy C API - required for using NumPy C functions
np.import_array()


cdef class CppList:
    cdef MKL_INT* data

    def __cinit__(self, size_t number):
        # allocate some memory (uninitialised, may contain arbitrary data)
        self.data = <MKL_INT*> PyMem_Malloc(
            number * sizeof(MKL_INT))
        if not self.data:
            raise MemoryError()

    def resize(self, size_t new_number):
        # Allocates new_number * sizeof(MKL_INT) bytes,
        # preserving the current content and making a best-effort to
        # re-use the original data location.
        mem = <MKL_INT*> PyMem_Realloc(
            self.data, new_number * sizeof(MKL_INT))
        if not mem:
            raise MemoryError()
        # Only overwrite the poMKL_INTer if the memory was really reallocated.
        # On error (mem is NULL), the originally memory has not been freed.
        self.data = mem

    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL


def blacs_pinfo():
    cdef MKL_INT rank, nprocs
    blacs_pinfo_(&rank, &nprocs)
    return rank, nprocs


def blacs_get(ictxt, what):
    cdef MKL_INT cwhat = what
    cdef MKL_INT cictxt = ictxt
    cdef MKL_INT cval
    blacs_get_(&cictxt, &cwhat, &cval)
    return cval


def blacs_gridinit(ictxt, layout, nprow, npcol):
    if layout not in ['C', 'R']:
        raise ValueError("layout must be 'C' or 'R'")
    cdef MKL_INT cictxt = ictxt
    cdef char clayout = ord(layout[0])  # Convert Python string to char
    cdef MKL_INT cnprow = nprow
    cdef MKL_INT cnpcol = npcol
    blacs_gridinit_(&cictxt, &clayout, &cnprow, &cnpcol)
    return cictxt


def blacs_gridmap(ictxt, usermap, ldumap, nprow, npcol):
    cdef MKL_INT cictxt = ictxt
    cusermap = CppList(nprow)
    for i, val in enumerate(usermap):
        cusermap.data[i] = val
    cdef MKL_INT cldumap = ldumap
    cdef MKL_INT cnprow = nprow
    cdef MKL_INT cnpcol = npcol
    blacs_gridmap_(&cictxt, cusermap.data, &cldumap, &cnprow, &cnpcol)
    return cictxt


def blacs_gridinfo(ictxt, nprow, npcol):
    cdef MKL_INT myrow, mycol
    cdef MKL_INT cictxt = ictxt
    cdef MKL_INT cnprow = nprow
    cdef MKL_INT cnpcol = npcol
    blacs_gridinfo_(&cictxt, &cnprow, &cnpcol, &myrow, &mycol)
    return cnprow, cnpcol, myrow, mycol


def blacs_gridexit(ictxt):
    cdef MKL_INT cictxt = ictxt
    blacs_gridexit_(&cictxt)
    return


def blacs_exit(cont):
    cdef MKL_INT ccont = cont
    blacs_exit_(&ccont)
    return


def blacs_pnum(ictxt, prow, pcol):
    cdef MKL_INT cictxt = ictxt
    cdef MKL_INT cprow = prow
    cdef MKL_INT cpcol = pcol
    cdef MKL_INT proc_num = blacs_pnum_(&cictxt, &cprow, &cpcol)
    return proc_num


def numroc(n, nb, iproc, nprocs, srcproc=0):
    cdef MKL_INT cn = n
    cdef MKL_INT cnb = nb
    cdef MKL_INT ciproc = iproc
    cdef MKL_INT csrcproc = srcproc
    cdef MKL_INT cnprocs = nprocs
    cdef MKL_INT numroc_info = numroc_(&cn, &cnb, &ciproc, &csrcproc, &cnprocs)
    return numroc_info


def ilcm(m, n):
    cdef MKL_INT cm = m
    cdef MKL_INT cn = n
    cdef MKL_INT ilcm_info = ilcm_(&cm, &cn)
    return ilcm_info


def indxg2p(indxglob, nb, iproc, nprocs):
    cdef MKL_INT cindxglob = indxglob
    cdef MKL_INT cnb = nb
    cdef MKL_INT ciproc = iproc
    cdef MKL_INT srcproc = 0
    cdef MKL_INT cnprocs = nprocs
    cdef MKL_INT indxg2p_info = indxg2p_(&cindxglob, &cnb, &ciproc, &srcproc, &cnprocs)
    return indxg2p_info


def descinit(m, n, mb, nb, ictxt, lld):
    """Initialize ScaLAPACK array descriptor.
    
    Parameters:
    m, n: Global dimensions of the matrix
    mb, nb: Block sizes for distribution
    ictxt: BLACS context
    lld: Local leading dimension (must be >= local number of rows)
    """
    cdef MKL_INT desc[9]
    cdef MKL_INT cm = m
    cdef MKL_INT cn = n
    cdef MKL_INT cmb = mb
    cdef MKL_INT cnb = nb
    cdef MKL_INT lrsrc = 0  # Source process row (usually 0)
    cdef MKL_INT lcsrc = 0  # Source process column (usually 0)
    cdef MKL_INT cictxt = ictxt
    cdef MKL_INT clld = max(1, lld)  # Ensure LLD is at least 1
    cdef MKL_INT info = 0

    descinit_(desc, &cm, &cn, &cmb, &cnb, &lrsrc, &lcsrc, &cictxt, &clld, &info)
    
    if info != 0:
        if info == -2:
            raise ValueError(f"DESCINIT error: M ({m}) < 0")
        elif info == -3:
            raise ValueError(f"DESCINIT error: N ({n}) < 0")
        elif info == -4:
            raise ValueError(f"DESCINIT error: MB ({mb}) < 1")
        elif info == -5:
            raise ValueError(f"DESCINIT error: NB ({nb}) < 1")
        elif info == -6:
            raise ValueError(f"DESCINIT error: IRSRC ({lrsrc}) not in [0, NPROW-1]")
        elif info == -7:
            raise ValueError(f"DESCINIT error: ICSRC ({lcsrc}) not in [0, NPCOL-1]")
        elif info == -8:
            raise ValueError(f"DESCINIT error: Invalid context (NPROW = -1)")
        elif info == -9:
            raise ValueError(f"DESCINIT error: LLD ({clld}) < local rows")
        else:
            raise ValueError(f"DESCINIT error: Parameter {-info} had an illegal value")
    
    return desc


def pdgels(m, n, rhs, A, descA, B, descB, X, maybe):
    cdef char trans = 'N'
    cdef MKL_INT aone = 1
    cdef MKL_INT bone = 1
    cdef MKL_INT info = 0
    cdef MKL_INT cm = m
    cdef MKL_INT cn = n
    cdef MKL_INT desc_A[9]
    cdef MKL_INT desc_B[9]
    for n, (a, b) in enumerate(zip(descA, descB)):
        desc_A[n] = a
        desc_B[n] = b
    cdef long lwork = maybe
    cdef long long yo = maybe
    cdef MKL_INT b_wid = rhs

    cdef np.ndarray[np.double_t, ndim=2, mode='fortran'] np_buffA = np.asfortranarray(A, dtype=np.double)
    cdef double* A_ptr = <double*> np.PyArray_DATA(np_buffA)

    cdef np.ndarray[np.double_t, ndim=2, mode='fortran'] np_buffB = np.asfortranarray(B, dtype=np.double)
    cdef double* B_ptr = <double*> np.PyArray_DATA(np_buffB)

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] np_buff_X = np.asfortranarray(X, dtype=np.double)
    cdef double* X_ptr = <double*> np.PyArray_DATA(np_buff_X)

    pdgels_(&trans, &cm, &cn, &b_wid, A_ptr, &aone, &aone, desc_A, B_ptr, &bone, &bone, desc_B, X_ptr, &lwork, &info)
    
    if info != 0:
        raise RuntimeError(f"PDGELS failed with INFO = {info}")


def lstsq(A, b, A_len, A_wid, num_nodes, temp):
    cdef MKL_INT nprow = num_nodes
    cdef MKL_INT npcol = 1
    cdef MKL_INT m = A_len
    cdef MKL_INT n = A_wid
    cdef MKL_INT mb = np.shape(A)[0]
    cdef MKL_INT nb = np.shape(A)[1]
    cdef MKL_INT myrow = 0
    cdef MKL_INT mycol = 0
    cdef MKL_INT descA[9]
    cdef MKL_INT descB[9]
    blacs_pinfo()
    cdef MKL_INT ictxt = blacs_get()
    ictxt = blacs_gridinit(ictxt, 'C', nprow, npcol)
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    cdef MKL_INT numroc_Ar = numroc(m, m, myrow, nprow)
    descA = descinit(m, n, mb, nb, ictxt, numroc_Ar)
    descB = descinit(m, 1, mb, 1, ictxt, numroc_Ar)
    cdef np.ndarray X = np.asfortranarray(np.zeros(temp), dtype=np.double)
    pdgels(m, n, 1, A, descA, b, descB, X, len(X))
    return b[:nb]


def pdgesvd(jobu, jobvt, m, n, A, ia, ja, descA, S, U, iu, ju, descU, VT, ivt, jvt, descVT, work, lwork):
    cdef char* cjobu = <bytes>jobu
    cdef char* cjobvt = <bytes>jobvt
    cdef MKL_INT cm = m
    cdef MKL_INT cn = n
    cdef MKL_INT cia = ia
    cdef MKL_INT cja = ja
    cdef MKL_INT ciu = iu
    cdef MKL_INT cju = ju
    cdef MKL_INT civt = ivt
    cdef MKL_INT cjvt = jvt
    cdef MKL_INT clwork = len(work)
    cdef MKL_INT info = 0
    cdef MKL_INT desc_A[9]
    cdef MKL_INT desc_U[9] 
    cdef MKL_INT desc_VT[9]

    for n, (a, u, vt) in enumerate(zip(descA, descU, descVT)):
        desc_A[n] = a
        desc_U[n] = u
        desc_VT[n] = vt

    cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buffA = np.ascontiguousarray(A, dtype=np.double)
    cdef double* A_ptr = <double*> np.PyArray_DATA(np_buffA)

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] np_buffS = np.ascontiguousarray(S, dtype=np.double)
    cdef double* S_ptr = <double*> np.PyArray_DATA(np_buffS)

    cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buffU = np.ascontiguousarray(U, dtype=np.double)
    cdef double* U_ptr = <double*> np.PyArray_DATA(np_buffU)

    cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buffVT = np.ascontiguousarray(VT, dtype=np.double)
    cdef double* VT_ptr = <double*> np.PyArray_DATA(np_buffVT)

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] np_buffwork = np.ascontiguousarray(work, dtype=np.double)
    cdef double* work_ptr = <double*> np.PyArray_DATA(np_buffwork)

    pdgesvd_(cjobu, cjobvt, &cm, &cn, A_ptr, &cia, &cja, desc_A, S_ptr, U_ptr, &ciu, &cju, desc_U, VT_ptr, &civt, &cjvt, desc_VT, work_ptr, &clwork, &info)



cdef class Scalapack:
    cdef MKL_INT ictxt
    cdef MKL_INT myrow
    cdef MKL_INT mycol
    cdef MKL_INT npcol
    cdef MKL_INT nprow
    cdef MKL_INT rzero
    cdef MKL_INT czero
    cdef MKL_INT myA_row
    cdef MKL_INT myA_col
    cdef MKL_INT a_len
    cdef MKL_INT a_wid
    cdef MKL_INT my_a_len
    cdef MKL_INT descA[9]
    cdef MKL_INT myB_row
    cdef MKL_INT myB_col
    cdef MKL_INT descB[9]
    cdef MKL_INT* usermap_ptr
    cdef np.ndarray usermap

    def __init__(self, num_nodes):
        self.nprow = num_nodes
        self.npcol = 1
        self.rzero = 0
        self.czero = 0
        self.usermap = np.ascontiguousarray(np.array([0, 2], dtype=np.int64), dtype=np.int64)
        self.usermap_ptr = <MKL_INT*> np.PyArray_DATA(self.usermap)
        self.initialize_blacs()

    def initialize_blacs(self):

        cdef MKL_INT rank
        cdef MKL_INT nprocs
        cdef MKL_INT zero = 0
        cdef char layout='R'

        # BLACS rank and world size
        blacs_pinfo_(&rank, &nprocs)

        # -> Create context
        blacs_get_(&zero, &zero, &self.ictxt )

        # Context -> Initialize the grid
        # blacs_gridinit_(&self.ictxt, &layout, &self.nprow, &self.npcol )

        # Context -> Initialize the grid
        blacs_gridmap_(&self.ictxt, self.usermap_ptr, &self.nprow, &self.nprow, &self.npcol )

        # Context -> Context grid info (# procs row/col, current procs row/col)
        blacs_gridinfo_(&self.ictxt, &self.nprow, &self.npcol, &self.myrow, &self.mycol )

    def compute_size(self, MKL_INT a_len, MKL_INT a_wid, MKL_INT my_a_len):

        self.a_len = a_len
        self.a_wid = a_wid
        self.my_a_len = my_a_len
        cdef MKL_INT b_wid = 1
        # Compute the size of the local matrices
        # My proc -> row of local A
        self.myA_row = numroc_( &self.a_len, &self.my_a_len, &self.myrow, &self.rzero, &self.nprow )
        # My proc -> col of local A
        self.myA_col = numroc_( &self.a_wid, &self.a_wid, &self.mycol, &self.czero, &self.npcol )
        # My proc -> row of local B
        # self.myB_row = numroc_( &self.a_len, &self.my_a_len, &self.myrow, &self.rzero, &self.nprow )
        # My proc -> col of local B
        # self.myB_col = numroc_( &b_wid, &b_wid, &self.mycol, &self.czero, &self.npcol )
        # print(
        #     "Hi. Proc {}/{} for MPI, proc {}/{} for BLACS in position"
        #     " ({},{})/({},{}) with local matrix {}x{}, global matrix {}, block size {}\n".format(
        #         myrank_mpi, nprocs_mpi, iam, nprocs, myrow, mycol, nprow, npcol, mpA, nqA, n, nb))
        # printf("%i %i\n", self.rzero, self.czero)
        return self.myA_row, self.myA_col, self.nprow, self.npcol, self.myrow, self.mycol

    def create_descriptor(self):

        cdef MKL_INT info_a
        cdef MKL_INT info_b
        cdef MKL_INT lddA

        if self.myA_row > 1:
            lddA = self.myA_row
        else:
            lddA = 1

        descinit_( self.descA, &self.a_len, &self.a_wid, &self.my_a_len, &self.a_wid, &self.rzero, &self.czero, &self.ictxt, &lddA, &info_a);
        if info_a != 0:
            print("Error in descinit, info = {}\n".format(info_a))

        # cdef MKL_INT lddB
        # if self.myB_row > 1:
        #     lddB = self.myB_row
        # else:
        #     lddB = 1

        cdef MKL_INT b_wid = 1
        descinit_( self.descB, &self.a_len, &b_wid, &self.my_a_len, &b_wid, &self.rzero, &self.czero, &self.ictxt, &lddA, &info_b);
        if info_b != 0:
            print("Error in descinit, info = {}\n".format(info_b))

        # assert info_a == info_b, "info_a must be equal to info_b"

    def dpotrf(self, A):
        # Run dpotrf and time
        # double MPIt1 = MPI_Wtime();
        # printf("[%dx%d] Starting potrf\n", myrow, mycol);
        cdef char uplo = 'L'
        cdef MKL_INT ione = 1
        cdef MKL_INT info = 0

        cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buff = np.ascontiguousarray(A, dtype=np.double)
        cdef double* A_ptr = <double*> np.PyArray_DATA(np_buff)
        pdpotrf_(&uplo, &self.a_len, A_ptr, &ione, &ione, self.descA, &info);

        # if info != 0:
        #     print("Error in potrf, info = {}\n".format(info));
        # double MPIt2 = MPI_Wtime();
        # printf("[%dx%d] Done, time %e s.\n", myrow, mycol, MPIt2 - MPIt1);

    def pdgels(self, A, B, X, maybe):
        cdef char trans = 'N'
        cdef MKL_INT aone = 1
        cdef MKL_INT bone = 1
        cdef MKL_INT info = 0
        cdef MKL_INT b_wid = 1

        cdef long lwork = maybe

        cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buffA = np.ascontiguousarray(A, dtype=np.double)
        cdef double* A_ptr = <double*> np.PyArray_DATA(np_buffA)

        cdef np.ndarray[np.double_t, ndim=2, mode='c'] np_buffB = np.ascontiguousarray(B, dtype=np.double)
        cdef double* B_ptr = <double*> np.PyArray_DATA(np_buffB)

        cdef np.ndarray[np.double_t, ndim=1, mode='c'] np_buffX = np.ascontiguousarray(X, dtype=np.double)
        cdef double* X_ptr = <double*> np.PyArray_DATA(np_buffX)

        pdgels_(&trans, &self.a_len, &self.a_wid, &b_wid, A_ptr, &aone, &aone, self.descA, B_ptr, &bone, &bone, self.descB, X_ptr, &lwork, &info)


def initialize_blacs(num_nodes):
    cdef MKL_INT rank = 0
    cdef MKL_INT nprocs = 0
    cdef MKL_INT zero = 0
    cdef MKL_INT ictxt, myrow, mycol
    cdef MKL_INT npcol = num_nodes
    cdef MKL_INT nprow = 1
    cdef char layout='R'

    # BLACS rank and world size
    blacs_pinfo_(&rank, &nprocs)

    # -> Create context
    blacs_get_(&zero, &zero, &ictxt )

    # Context -> Initialize the grid
    # blacs_gridinit_(&ictxt, &layout, &nprow, &npcol )
    cdef MKL_INT[2] X_ptr = {0, 2}
    blacs_gridmap_(&ictxt, X_ptr, &nprow, &nprow, &npcol )

    # Context -> Context grid info (# procs row/col, current procs row/col)
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol )

    return rank, nprocs, ictxt, myrow, mycol

# def compute_size(MKL_INT a_len, MKL_INT a_wid, MKL_INT my_a_len, myrow, mycol, izero, nprow, npcol):
#
#     # Compute the size of the local matrices
#     # My proc -> row of local A
#     MKL_INT myA_row    = numroc_( &a_len, &my_a_len, &myrow, &izero, &nprow )
#     # My proc -> col of local A
#     MKL_INT myA_col    = numroc_( &a_wid, &a_wid, &mycol, &izero, &npcol )
#
#     return myA_row, myA_col


