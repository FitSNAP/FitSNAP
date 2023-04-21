from fitsnap3lib.lib.sym_ACE.gen_labels import *
from fitsnap3lib.lib.sym_ACE.coupling_coeffs import *

#library to generate generalized Wigner symbols using sigma_c symmetric (full ordered) 
#  binary trees.

def get_ms(l,M_R=0):
    # retrieves the set of m_i combinations obeying \sum_i m_i = 0 for an arbitrary l vector
    m_ranges={ind:range(-l[ind],l[ind]+1) for ind in range(len(l))}
    m_range_arrays = [list(m_ranges[ind]) for ind in range(len(l))]
    m_combos = list(itertools.product(*m_range_arrays))
    first_m_filter = [i for i in m_combos if np.sum(i) == M_R]
    m_list_replace = ['%d']*len(l)
    m_str_variable = ','.join(b for b in m_list_replace)
    m_strs = [ m_str_variable % fmf for fmf in first_m_filter]
    return m_strs

def rank_1_tree(l,Wigner_3j,L_R=0,M_R=0):
    #no nodes for rank 1

    mstrs = get_ms(l,M_R)
    full_inter_tuples = [()]
    assert l[0] == L_R, "invalid l=%d for irrep L_R = %d" % (l[0],L_R)
    
    decomposed = {full_inter_tup:{mstr:0.0 for mstr in mstrs} for full_inter_tup in full_inter_tuples}

    for inter in full_inter_tuples:
        for mstr in mstrs:
            m_ints = [int(b) for b in mstr.split(',')]
            # m_1  = - M_R
            conds= m_ints[0]  == - M_R
            if conds:
                w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],L_R,M_R,0,0)]
                phase_power = 0
                phase = (-1) ** phase_power
                w = phase * w1

                decomposed[inter][mstr] = float(w)
    return decomposed


def rank_2_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L_R,-M_R)]
                phase_power = L_R - M_R
                phase = (-1) ** phase_power
                w = phase * w1 

                decomposed[inter][mstr] = (w)
    return decomposed

def rank_3_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,-M1)]
                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1,M1,l[2],m_ints[2],L_R,-M_R)]
                    phase_power = ( L1 ) - ( M1  )  + ( L_R - M_R)
                    phase = (-1) ** phase_power
                    w = phase * w1 * w2
                    decomposed[inter][mstr] = (w)
    return decomposed
    

def rank_4_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                        w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,-M1)]
                        w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[2],m_ints[2],l[3],m_ints[3],L2,-M2)]
                        w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1,M1,L2,M2,L_R,-M_R)]
                        phase_power = ( L1 + L2 ) - ( M1 + M2 )  + ( L_R - M_R)
                        phase = (-1) ** phase_power
                        w = phase * w1 * w2 * w3 

                        decomposed[inter][mstr] = (w)
    return decomposed



def rank_5_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                            w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0],m_ints[0],l[1],m_ints[1],L1,-M1)]
                            w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[2],m_ints[2],l[3],m_ints[3],L2,-M2)]
                            w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1,M1,L2,M2,L3,-M3)]
                            w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L3,M3,l[4],m_ints[4],L_R,-M_R)]
                            phase_power = ( L1 + L2 + L3 ) - ( M1 + M2 + M3 )  + ( L_R - M_R)
                            phase = (-1) ** phase_power
                            w = phase * w1 * w2 * w3 * w4

                            decomposed[inter][mstr] = (w)
    return decomposed

def rank_6_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                                w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , -M1)]
                                w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , -M2)]
                                w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , -M3)]
                                w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L4 , -M4)]
                                w5 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L3 , M3 , L4 , M4 , L_R , -M_R)]
                                phase_power = ( L1 + L2 + L3 + L4 ) - ( M1 + M2 + M3 + M4 )  + ( L_R - M_R)
                                phase = (-1) ** phase_power
                                w = phase * w1 * w2 * w3 * w4 * w5

                                decomposed[inter][mstr] = (w)
    return decomposed

def rank_7_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                                    w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , -M1)]
                                    w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , -M2)]
                                    w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , -M3)]
                                    w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L4 , -M4)]
                                    w5 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L3 , M3 , l[6] , m_ints[6] , L5 , -M5)]
                                    w6 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L4 , M4 , L5 , M5 , L_R , -M_R)]
                                    phase_power = ( L1 + L2 + L3 + L4 + L5 ) - ( M1 + M2 + M3 + M4 + M5 )  + ( L_R - M_R)
                                    phase = (-1) ** phase_power
                                    w = phase * w1 * w2 * w3 * w4 * w5 * w6

                                    decomposed[inter][mstr] = (w)
    return decomposed

def rank_8_tree(l,Wigner_3j,L_R=0,M_R=0):

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
                                        w1 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[0] , m_ints[0] , l[1] , m_ints[1] , L1 , -M1)]
                                        w2 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[2] , m_ints[2] , l[3] , m_ints[3] , L2 , -M2)]
                                        w3 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[4] , m_ints[4] , l[5] , m_ints[5] , L3 , -M3)]
                                        w4 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(l[6] , m_ints[6] , l[7] , m_ints[7] , L4 , -M4)]
                                        w5 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L1 , M1 , L2 , M2 , L5 , -M5)]
                                        w6 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L3 , M3 , L4 , M4 , L6 , -M6)]
                                        w7 = Wigner_3j['%d,%d,%d,%d,%d,%d' %(L5 , M5 , L6 , M6 , L_R , -M_R)]
                                        phase_power = ( L1 + L2 + L3 + L4 + L5 + L6 ) - ( M1 + M2 + M3 + M4 + M5 + M6 )  + ( L_R - M_R)
                                        phase = (-1) ** phase_power
                                        w = phase * w1 * w2 * w3 * w4 * w5 * w6 * w7

                                        decomposed[inter][mstr] = float(w)
    return decomposed
