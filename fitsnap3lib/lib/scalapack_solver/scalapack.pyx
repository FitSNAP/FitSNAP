# distutils: language = c++

from Scalapack cimport *
from libc.stdlib cimport calloc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np


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
    # if layout == 'C' and layout != 'R':
    #     raise ValueError("layout must be C or R")
    cdef MKL_INT cictxt = ictxt
    # cdef char* clayout = <bytes>layout
    cdef char clayout = 'R'
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


def descinit(m, n, mb, nb, ictxt, numroc_info):
    cdef MKL_INT desc[9]
    cdef MKL_INT cm = m
    cdef MKL_INT cn = n
    cdef MKL_INT cmb = mb
    cdef MKL_INT cnb = nb
    cdef MKL_INT lrsrc = 0
    cdef MKL_INT lcsrc = 0
    cdef MKL_INT cictxt = ictxt
    cdef MKL_INT lddA
    if numroc_info > 1:
        lddA = numroc_info
    else:
        lddA = 1
    cdef MKL_INT info

    descinit_( desc, &cm, &cn, &cmb, &cnb, &lrsrc, &lcsrc, &cictxt, &lddA, &info)
    if info != 0:
            print("Error in descinit, info = {}\n".format(info))
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

    cdef np.ndarray np_buffA = np.asfortranarray(A, dtype=np.double)
    cdef double* A_ptr = <double*> np_buffA.data

    cdef np.ndarray np_buffB = np.asfortranarray(B, dtype=np.double)
    cdef double* B_ptr = <double*> np_buffB.data

    cdef np.ndarray np_buff_X = np.asfortranarray(X, dtype=np.double)
    cdef double* X_ptr = <double*> np_buff_X.data
    print(maybe, lwork, yo)

    pdgels_(&trans, &cm, &cn, &b_wid, A_ptr, &aone, &aone, desc_A, B_ptr, &bone, &bone, desc_B, X_ptr, &lwork, &info)


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
    cdef MKL_INT desc_A[9], desc_U[9], desc_VT[9]

    for n, (a, u, vt) in enumerate(zip(descA, descU, descVT)):
        desc_A[n] = a
        desc_U[n] = u
        desc_VT[n] = vt

    cdef np.ndarray np_buffA = np.ascontiguousarray(A, dtype=np.double)
    cdef double* A_ptr = <double*> np_buffA.data

    cdef np.ndarray np_buffS = np.ascontiguousarray(S, dtype=np.double)
    cdef double* S_ptr = <double*> np_buffS.data

    cdef np.ndarray np_buffU = np.ascontiguousarray(U, dtype=np.double)
    cdef double* U_ptr = <double*> np_buffU.data

    cdef np.ndarray np_buffVT = np.ascontiguousarray(VT, dtype=np.double)
    cdef double* VT_ptr = <double*> np_buffVT.data

    cdef np.ndarray np_buffwork = np.ascontiguousarray(work, dtype=np.double)
    cdef double* work_ptr = <double*> np_buffwork.data

    pdgesvd_(cjobu, cjobvt, &cm, &cn, A_ptr, &cia, &cja, desc_A, S_ptr, U_ptr, &ciu, &cju, desc_U, VT_ptr, &civt, &cjvt, desc_VT, work_ptr, &clwork, &info)



cdef class Scalapack:
    cdef MKL_INT ictxt, myrow, mycol, npcol, nprow, rzero, czero, myA_row, myA_col, a_len, a_wid, my_a_len, descA[9]
    cdef MKL_INT myB_row, myB_col, descB[9]
    cdef MKL_INT* usermap_ptr
    cdef np.ndarray usermap

    def __init__(self, num_nodes):
        self.nprow = num_nodes
        self.npcol = 1
        self.rzero = 0
        self.czero = 0
        self.usermap = np.ascontiguousarray(np.array([0, 2], dtype=np.MKL_INT), dtype=np.MKL_INT)
        self.usermap_ptr = <MKL_INT*> self.usermap.data
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

        cdef np.ndarray np_buff = np.ascontiguousarray(A, dtype=np.double)
        cdef double* A_ptr = <double*> np_buff.data
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

        cdef np.ndarray np_buffA = np.ascontiguousarray(A, dtype=np.double)
        cdef double* A_ptr = <double*> np_buffA.data

        cdef np.ndarray np_buffB = np.ascontiguousarray(B, dtype=np.double)
        cdef double* B_ptr = <double*> np_buffB.data

        cdef np.ndarray np_buffX = np.ascontiguousarray(X, dtype=np.double)
        cdef double* X_ptr = <double*> np_buffX.data

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


