from scalapack_funcs import *
import ctypes as ctypes
from fitsnap3lib.parallel_tools import ParallelTools


pt = ParallelTools()


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


def lstsq(A, b, lengths=None):
    # TODO: calculate size of work matrix needed
    if lengths is None:
        raise ValueError("lengths should not be none!")
    else:
        total_length, node_length, scraped_length = lengths
    nprow = pt.get_number_of_nodes()
    npcol = 1
    mb = int(np.floor(total_length / nprow))
    m = total_length
    n = np.shape(A)[-1]
    nb = n
    blacs_pinfo()

    ictxt = blacs_get(0, 0)
    ictxt = blacs_gridmap(ictxt, pt.get_node_list(), nprow, nprow, npcol)
    # nprow = Number of process rows in the current process grid.
    # npcol = Number of process columns in the current process grid.
    # myprow = Row coordinate of the calling process in the process grid.
    # mypcol = Column coordinate of the calling process in the process grid.
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    numroc_Ar = numroc(m, mb, myrow, nprow)
    descA = descinit(m, n, mb, nb, ictxt, numroc_Ar)
    descB = descinit(m, 1, mb, 1, ictxt, numroc_Ar)
    upper = node_length
    A = A[:upper]
    b = b[:upper]
    work_length = find_work_space(myrow, mycol, nprow, npcol, m, mb, n)
    work = np.zeros(work_length)
    pdgels(m, n, 1, A, descA, b, descB, work, work_length)
    if pt.get_rank() == 0:
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        b = np.ctypeslib.as_array(b_ptr,shape=(n,))
    blacs_gridexit(ictxt)
    pt.all_barrier()
    return b


def dummy_lstsq():
    nprow = pt.get_number_of_nodes()
    npcol = 1
    blacs_pinfo()

    ictxt = blacs_get(0, 0)
    ictxt = blacs_gridmap(ictxt, pt.get_node_list(), nprow, nprow, npcol)
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    pt.all_barrier()
    return None
