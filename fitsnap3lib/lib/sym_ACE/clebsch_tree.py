from fitsnap3lib.lib.sym_ACE.gen_labels import *
from fitsnap3lib.lib.sym_ACE.coupling_coeffs import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm = comm)


@pt.rank_zero
def get_ms(l,M_R=0):

    # retrieves the set of m_i combinations obeying \sum_i m_i = M_R for an arbitrary l vector
    m_ranges={ind:range(-l[ind],l[ind]+1) for ind in range(len(l))}
    m_range_arrays = [list(m_ranges[ind]) for ind in range(len(l))]
    m_combos = list(itertools.product(*m_range_arrays))
    first_m_filter = [i for i in m_combos if np.sum(i) == M_R]
    m_list_replace = ['%d']*len(l)
    m_str_variable = ','.join(b for b in m_list_replace)
    m_strs = [ m_str_variable % fmf for fmf in first_m_filter]
    return m_strs

#manually coded reductions of spherical harmonics 
@pt.rank_zero
def rank_1_cg_tree(l,L_R=0,M_R=0):

    mstrs = get_ms(l,M_R)
    full_inter_tuples = [()]
    assert l[0] == L_R, "invalid l=%d for irrep L_R = %d" % (l[0],L_R)
    
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            # m_1  = - M_R
            conds= m_ints[0]  ==  M_R
            if conds:
                w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],L_R,M_R,0,0)]
                phase = 1 
                w = phase * w1

                decomposed[inter][mstr] = w
    return decomposed


@pt.rank_zero
def rank_2_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    node = nodes[0]
    mstrs = get_ms(l,M_R)
    full_inter_tuples = [()]

    assert check_triangle(l[0],l[1],L_R) , "invalid l=(%d,%d) for irrep L_R = %d" % (l[0],l[1],L_R)

    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            # m_1 + m_2 = M1
            conds= (m_ints[0] + m_ints[1]) == M_R
            if conds:
                w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L_R,M_R)]
                phase = 1
                w = phase * w1 

                decomposed[inter][mstr] = (w)
    return decomposed

@pt.rank_zero
def rank_3_cg_tree(l,L_R=0,M_R=0):

    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    mstrs = get_ms(l,M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1 = inter[0]
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1,L1+1):
                # m_1 + m_2 = M1
                # M1 + m_3 = M_R
                conds= (m_ints[0] + m_ints[1]) == M1 and\
                (M1+m_ints[2]) == M_R
                if conds:
                    w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,M1)]
                    w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1,M1,l[2],m_ints[2],L_R,M_R)]
                    phase = 1
                    w = phase * w1 * w2
                    decomposed[inter][mstr] = (w)
    return decomposed


@pt.rank_zero
def generalized_rank_4_inters(grouped_l,L_R):
    l = flatten(grouped_l)
    l1,l2,l3,l4 = tuple(l)
    size_by_orbit = { io: len(o) for io,o in enumerate(grouped_l) }
    pairs = [p for p in grouped_l if len(p) ==2]
    singles =[ p for p in grouped_l if len(p) < 2]
    assert len(pairs + singles) == len(grouped_l), "bad orbit size for intermediate vector"

    base_inters = {}
    valid_inters = []
    mstrs = []
    for io,o in enumerate(grouped_l):
        if size_by_orbit[io] == 2:
            base_inters[io] = [Lk for Lk in get_intermediates_w(o[0],o[1])  if np.sum(o + (Lk,)) % 2 ==0 ] 
    if len(base_inters.keys()) == 2:
        inters_set = [ (bi1,bi2) for bi1,bi2 in itertools.product( base_inters[0] , base_inters[1]) if check_triangle(bi1,bi2,L_R)]
        valid_inters.extend(inters_set)
    elif len(base_inters.keys()) == 1:
        size_2 = list(base_inters.keys())[0]
        for inter1 in base_inters[size_2]:
            
            inters_next = [ i for i in get_intermediates_w(inter1, grouped_l[1][0]) if check_triangle(grouped_l[2][0],i,L_R)]
            current_valids = [(inter1,inxt) for inxt in inters_next]
            valid_inters.extend(current_valids)
    """
    for inter in valid_inters:
        o1 = grouped_l[0]
        o2 = grouped_l[1]
        o1prd = itertools.product(range(-o1[0],o1[0]+1),range(-o1[1],o1[1]+1))
        o2prd = itertools.product(range(-o2[0],o2[0]+1),range(-o2[1],o2[1]+1))
        for Mo1 in range(-inter[0],inter[0]+1):
            mo1s = [ (mi,mj) for (mi,mj) in o1prd if np.sum((mi,mj)) == Mo1 ] 
    """
    return valid_inters
        
"""
def generalized_rank_4_cg_tree(l,sigma_c,L_R=0,M_R=0):
    print ('NOTE the m vectors in this loop are only generalized for COMPLETELY degenerate l_i')
    grouped_l = group_vec_by_orbits(l,sigma_c)

    mstrs = get_ms(l,M_R)
    full_inter_tuples = generalized_rank_4_inters(grouped_l,L_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1,L2 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1,L1+1):
                for M2 in range(-L2,L2+1):
                    conds_per_couple_typ = { (2,2): (m_ints[0] + m_ints[1]) == M1 and\
                                    (m_ints[2] + m_ints[3]) == M2 and\
                                    (M1+M2) == M_R,
                                   (2,1,1): (m_ints[0] + m_ints[1]) == M1 and\
                                    (m_ints[2] + M1) == M2 and\
                                    (m_ints[3] + M2) == M_R
                                }
                    conds = conds_per_couple_typ[sigma_c]
                    if conds:
                        if sigma_c == (2,2):
                            w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,M1)]
                            w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2],m_ints[2],l[3],m_ints[3],L2,M2)]
                            w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1,M1,L2,M2,L_R,M_R)]
                            phase = 1
                            w = phase * w1 * w2 * w3
                            decomposed[inter][mstr] = (w)
                        elif sigma_c == (2,1,1):
                            w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,M1)]
                            w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1,M1,l[2],m_ints[2],L2,M2)]
                            w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L2,M2,l[3],m_ints[3],L_R,M_R)]
                            phase = 1
                            w = phase * w1 * w2 * w3
                            decomposed[inter][mstr] = (w)
    return decomposed    

ltst = [1,1,1,1]
#sigmactst = (2,1,1)
sigmactst = (2,2)
these_cgs = generalized_rank_4_cg_tree(l=ltst,sigma_c=sigmactst,L_R=0,M_R=0)

print (these_cgs)
"""
@pt.rank_zero
def rank_4_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    mstrs = get_ms(l,M_R)
    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1,L2 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1,L1+1):
                for M2 in range(-L2,L2+1):
                    # m_1 + m_2 = M1
                    # m_4 + m_3 = M2
                    # M1 + M2 = M_R
                    conds= (m_ints[0] + m_ints[1]) == M1 and\
                    (m_ints[2] + m_ints[3]) == M2 and\
                    (M1+M2) == M_R
                    if conds:
                        w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,M1)]
                        w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2],m_ints[2],l[3],m_ints[3],L2,M2)]
                        w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1,M1,L2,M2,L_R,M_R)]
                        #phase_power = ( L1 + L2 ) - ( M1 + M2 )  + ( L_R - M_R)
                        #phase = (-1) ** phase_power
                        phase = 1
                        w = phase * w1 * w2 * w3 

                        decomposed[inter][mstr] = (w)
    return decomposed



@pt.rank_zero
def rank_5_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    mstrs = get_ms(l,M_R)
    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1,L2,L3 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1,L1+1):
                for M2 in range(-L2,L2+1):
                    for M3 in range(-L3,L3+1):
                        # m_1 + m_2 = M1
                        # m_4 + m_3 = M2
                        # M1 + M2 = M3
                        conds= (m_ints[0] + m_ints[1]) == M1 and\
                        (m_ints[2] + m_ints[3]) == M2 and\
                        (M1+M2) == M3 and\
                        (M3+m_ints[4]) == M_R
                        if conds:
                            w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,M1)]
                            w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2],m_ints[2],l[3],m_ints[3],L2,M2)]
                            w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1,M1,L2,M2,L3,M3)]
                            w4 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L3,M3,l[4],m_ints[4],L_R,M_R)]
                            phase = 1
                            w = phase * w1 * w2 * w3 * w4

                            decomposed[inter][mstr] = (w)
    return decomposed

@pt.rank_zero
def rank_6_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    mstrs = get_ms(l,M_R)
    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1 , L2 , L3 , L4 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1 , L1 + 1):
                for M2 in range(-L2 , L2 + 1 ):
                    for M3 in range(-L3 , L3 + 1):
                        for M4 in range(-L4 , L4 + 1):
                            # m_1 + m_2 = M1
                            # m_4 + m_3 = M2
                            # m_5 + m_6 = M3
                            # M1 + M2 = M4
                            # M3 + M4 = M_R
                            conds= (m_ints[0] + m_ints[1]) == M1 and\
                            (m_ints[2] + m_ints[3]) == M2 and\
                            (m_ints[4] + m_ints[5]) == M3 and\
                            ( M1 + M2 ) == M4 and\
                            ( M3 + M4 ) == M_R
                            if conds:
                                w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , M1)]
                                w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , M2)]
                                w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , M3)]
                                w4 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L4 , M4)]
                                w5 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L3 , M3 , L4 , M4 , L_R , M_R)]
                                phase = 1
                                w = phase * w1 * w2 * w3 * w4 * w5

                                decomposed[inter][mstr] = (w)
    return decomposed

@pt.rank_zero
def rank_7_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    mstrs = get_ms(l,M_R)
    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1 , L2 , L3 , L4 , L5 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1 , L1 + 1):
                for M2 in range(-L2 , L2 + 1 ):
                    for M3 in range(-L3 , L3 + 1):
                        for M4 in range(-L4 , L4 + 1):
                            for M5 in range(-L5 , L5 + 1):
                                # m_1 + m_2 = M1
                                # m_4 + m_3 = M2
                                # m_5 + m_6 = M3
                                # M1 + M2 = M4
                                # M3 + m_7 = M5
                                # M3 + M4 = M_R
                                conds= (m_ints[0] + m_ints[1]) == M1 and\
                                (m_ints[2] + m_ints[3]) == M2 and\
                                (m_ints[4] + m_ints[5]) == M3 and\
                                ( M1 + M2 ) == M4 and\
                                ( M3 + m_ints[6]) == M5 and\
                                ( M4 + M5 ) == M_R
                                if conds:
                                    w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , M1)]
                                    w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , M2)]
                                    w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , M3)]
                                    w4 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L4 , M4)]
                                    w5 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L3 , M3 , l[6] , m_ints[6] , L5 , M5)]
                                    w6 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L4 , M4 , L5 , M5 , L_R , M_R)]
                                    phase = 1
                                    w = phase * w1 * w2 * w3 * w4 * w5 * w6

                                    decomposed[inter][mstr] = (w)
    return decomposed

@pt.rank_zero
def rank_8_cg_tree(l,L_R=0,M_R=0):

    nodes,remainder = tree(l)
    mstrs = get_ms(l,M_R)
    full_inter_tuples = tree_l_inters(l,L_R=L_R,M_R=M_R)
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        L1 , L2 , L3 , L4 , L5 , L6 = inter
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            for M1 in range(-L1 , L1 + 1):
                for M2 in range(-L2 , L2 + 1 ):
                    for M3 in range(-L3 , L3 + 1):
                        for M4 in range(-L4 , L4 + 1):
                            for M5 in range(-L5 , L5 + 1):
                                for M5 in range(-L6 , L6 + 1):
                                    # m_1 + m_2 = M1
                                    # m_4 + m_3 = M2
                                    # m_5 + m_6 = M3
                                    # M1 + M2 = M5
                                    # M3 + M4 = M6
                                    # M5 + M6 = M_R
                                    conds= (m_ints[0] + m_ints[1]) == M1 and\
                                    (m_ints[2] + m_ints[3]) == M2 and\
                                    (m_ints[4] + m_ints[5]) == M3 and\
                                    ( M1 + M2 ) == M5 and\
                                    ( M3 + M4) == M6 and\
                                    ( M5 + M6 ) == M_R
                                    if conds:
                                        w1 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , M1)]
                                        w2 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , M2)]
                                        w3 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , M3)]
                                        w4 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(l[6] , m_ints[6] , l[7] , m_ints[7] , L4 , M4)]
                                        w5 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L5 , M5)]
                                        w6 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L3 , M3 , L4 , M4 , L6 , M6)]
                                        w7 = Clebsch_Gordan['%d,%d,%d,%d,%d,%d' %(L5 , M5 , L6 , M6 , L_R , M_R)]
                                        phase = 1
                                        w = phase * w1 * w2 * w3 * w4 * w5 * w6 * w7

                                        decomposed[inter][mstr] = w
    return decomposed

del pt
