from scalapack_funcs import *
from ...parallel_tools import pt


def find_work_space(myrow, mycol, nprow, npcol, m, mb_a, n):
    ja, nb_a, nb_b = 1, n, 1
    mb_b = nb_a
    ltau = numroc(ja+n-1, nb_a, mycol, npcol)
    iroffa = 0 % mb_a
    icoffa = 0 % nb_a
    # iarow = indxg2p(1, mb_a, myrow, nprow)
    # iacol = indxg2p(1, nb_a, mycol, npcol)
    mpa0 = numroc(m+iroffa, mb_a, myrow, nprow)
    nqa0 = numroc(n+icoffa, nb_a, mycol, npcol)
    iroffb = 0 % nb_a
    icoffb = 0 % nb_b
    # ibrow = indxg2p(1, mb_b, myrow, nprow)
    # ibcol = indxg2p(1, nb_b, mycol, npcol)
    npb0 = numroc(n+iroffb, mb_b, myrow, nprow)
    nrhsqb0 = numroc(1+icoffb, nb_b, mycol, npcol)
    lwf = nb_a * (mpa0 + nqa0 + nb_a)
    lws = max(int((nb_a * (nb_a-1))/2), ((nrhsqb0+npb0)*nb_a)+nb_a*nb_a)
    work_length = ltau + max(lwf, lws)
    return work_length+1


def lstsq(A, b, lengths=None):
    # TODO: calculate size of work matrix needed
    if lengths is None:
        raise ValueError("lengths should not be none!")
    else:
        total_length, node_length, proc_length, scraped_length = lengths
    nprow = pt.get_size()
    rank = pt.get_rank()
    node = pt.get_node()
    subsize = pt.get_subsize()
    subrank = pt.get_subrank()
    npcol = 1
    mb = proc_length
    # m = total length of A
    # n = total width of A
    m = total_length
    n = np.shape(A)[-1]
    nb = n
    blacs_pinfo()
    ictxt = blacs_get()
    ictxt = blacs_gridinit(ictxt, 'C', nprow, npcol)
    # nprow = Number of process rows in the current process grid.
    # npcol = Number of process columns in the current process grid.
    # myprow = Row coordinate of the calling process in the process grid.
    # mypcol = Column coordinate of the calling process in the process grid.
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt, nprow, npcol)
    numroc_Ar = numroc(m, mb, myrow, nprow)
    descA = descinit(m, n, mb, nb, ictxt, numroc_Ar)
    descB = descinit(m, 1, mb, 1, ictxt, numroc_Ar)
    diff = 0
    if node == 0:
        diff = int(m - mb * nprow)
    lower = int((subrank/subsize)*(proc_length*subsize))
    # lower = int((rank/nprow)*(mb*nprow))
    upper = int(((subrank+1)/subsize)*(proc_length*subsize))
    # upper = int(((rank+1)/nprow)*(mb*nprow))
    if rank != 0:
        lower += diff
    upper += diff
    A = A[lower:upper]
    b = b[lower:upper]
    work_length = find_work_space(myrow, mycol, nprow, npcol, m, mb, n)
    work = np.zeros(work_length)
    pdgels(m, n, 1, A, descA, b, descB, work, work_length)
    return b[:n]
