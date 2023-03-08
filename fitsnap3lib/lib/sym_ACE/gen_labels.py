import itertools
import numpy as np
import math
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import *
from collections import Counter

# library of useful functions for generating labels
# (including lexicographical label generation)
# the bulk of this file is manual construction of l_vectors and intermediates

def group_vec_by_orbits(vec,part):
    ind_range = np.sum(part)
    assert len(vec) == ind_range, "vector must be able to fit in the partion"
    count = 0
    by_orbits = []
    for orbit in part:
        orbit_vec = []
        for i in range(orbit):
            orbit_vec.append(vec[count])
            count +=1
        by_orbits.append(tuple(orbit_vec))
    return tuple(by_orbits)

def group_vec_by_node(vec , nodes , remainder=None):
    #vec_by_tups = [tuple([vec[node[0]],vec[node[1]]]) for node in nodes]
    vec_by_tups = []
    for node in nodes:
        orbit_list = []
        for inode in node:
            orbit_list.append(vec[inode])
        orbit_tup = tuple(orbit_list)
        vec_by_tups.append(orbit_tup)
    if remainder != None:
        vec_by_tups = vec_by_tups + [tuple([vec[remainder]])]
    return vec_by_tups

def flatten(lstoflsts):
    try:
        flat = [i for sublist in lstoflsts for i in sublist]
        return flat
    except TypeError:
        return lstoflsts

def sort_pair(l):
    uniques = sorted(list(set(l)))
    per_unique = {u:[] for u in uniques}
    for li in l:
        per_unique[li].append(li)
    unsorted_tups = []
    for lu in uniques:
        countu = l.count(lu)
        if countu %2 ==0:
            nd = int(countu/2)
            resid = 0
        elif countu %2 !=0:
            nd = math.floor(countu/2)
            resid = 1
        pairls = [tuple([lu]*2)]*nd
        residls = [tuple([lu])]*resid
        unsorted_tups.append(pairls)
        unsorted_tups.append(residls)
    tups = sorted(flatten(unsorted_tups))
    tups.sort(key = lambda x : len(x),reverse = True)
    resorted = flatten(tups)
    return resorted

def srt_by_attyp(nulst):
    mu0s = []
    for nu in nulst:
        mu0 = nu.split('_')[0]
        if mu0 not in mu0s:
            mu0s.append(mu0)
    mu0s = sorted(mu0s)
    byattyp = {mu0:[] for mu0 in mu0s}
    for nu in nulst:
        mu0 = nu.split('_')[0]
        byattyp[mu0].append(nu)
    return byattyp

def get_mu_nu_rank(nu_in):
    if len(nu_in.split('_')) > 1:
        assert len(nu_in.split('_')) <= 3, "make sure your descriptor label is in proper format: mu0_mu1,mu2,mu3,n1,n2,n3,l1,l2,l3_L1" 
        nu = nu_in.split('_')[1]
        nu_splt = nu.split(',')
        return int(len(nu_splt)/3)
    else:
        nu = nu_in
        nu_splt = nu.split(',')
        return int(len(nu_splt)/2)

def get_mu_n_l(nu_in, return_L = False, **kwargs):
    rank = get_mu_nu_rank(nu_in)
    if len(nu_in.split('_')) > 1:
        if len(nu_in.split('_')) == 2:
            nu = nu_in.split('_')[-1]
            Lstr = ''
        else:
            nu = nu_in.split('_')[1]
            Lstr = nu_in.split('_')[-1]
        mu0 = int(nu_in.split('_')[0])
        nusplt = [int(k) for k in nu.split(',')]
        mu = nusplt[:rank]
        n = nusplt[rank:2*rank]
        l = nusplt[2*rank:]
        if len(Lstr) >= 1:
            L = [int(k) for k in Lstr.split('-')]
        else:
            L = None
        if return_L:
            return mu0 , mu , n , l , L
        else:
            return mu0 , mu , n , l
    #provide option to get n,l for depricated descriptor labels
    else:
        nu = nu_in
        mu0 = 0
        mu = [0]*rank
        nusplt = [int(k) for k in nu.split(',')]
        n = nusplt[:rank]
        l = nusplt[rank:2*rank]
        return mu0,mu,n,l

def ind_vec(lrng , size):
    uniques = []
    combs = itertools.combinations_with_replacement(lrng,size)
    for comb in combs:
        perms = itertools.permutations(comb)
        for p in perms:
            pstr = ','.join(str(k) for k in p)
            if pstr not in uniques:
                uniques.append(pstr)
    return uniques

def get_nu_rank(nu):
    nu_splt = [int(k) for k in nu.split(',')]
    if len(nu_splt) == 3:
        if nu_splt[1] == 0 and nu_splt[2] == 0:
            return 1
        elif nu_splt[1] != 0 or nu_splt[2] != 0:
            return 2
    elif len(nu_splt) > 3:
        return int(len(nu_splt) / 2)

def check_triangle(l1,l2,l3):
    #checks triangle condition between |l1+l2| and l3
    lower_bound = np.abs(l1 - l2)
    upper_bound = np.abs(l1 + l2)
    condition = l3 >= lower_bound and l3 <= upper_bound
    return condition

def get_intermediates(l):
    try:
        l = l.split(',')
        l1 = int(l[0])
        l2 = int(l[1])
    except AttributeError:
        l1 = l[0]
        l2 = l[1]

    tris = [i for i in range(abs(l1-l2),l1+l2+1)]

    ints = [i for i in tris]
    return ints

def unique_perms(vec):
    all_perms = [p for p in itertools.permutations(vec)]
    return sorted(list(set(all_perms)))

#wrapper
def get_intermediates_w(l1,l2):
    l = [l1,l2]
    return get_intermediates(l)

def tree(l):
    # quick construction of tree leaves
    rank = len(l)
    rngs = list(range(0,rank))
    rngs = iter(rngs)
    count = 0
    tup = []
    while count < int(rank/2):
        c1 = next(rngs)
        c2 = next(rngs)
        tup.append((c1,c2))
        count +=1
    remainder = None
    if rank %2 != 0:
        remainder = list(range(rank))[-1]
    return tuple(tup),remainder


def vec_nodes(vec,nodes,remainder=None):
    vec_by_tups = [tuple([vec[node[0]],vec[node[1]]]) for node in nodes]
    if remainder != None:
        vec_by_tups = vec_by_tups 
    return vec_by_tups

def tree_l_inters(l,L_R=0,M_R=0):
    # this function calculates the intermediates for l vectors of certain lengths
    nodes,remainder = tree(l)
    rank = len(l)
    if rank >=3:
        base_node_inters = {node:get_intermediates_w(l[node[0]],l[node[1]]) for node in nodes}

    full_inter_tuples = []

    if rank == 1:
        full_inter_tuples.append(())
    elif rank == 2:
        full_inter_tuples.append(())
    elif rank == 3:
        L1s = [i for i in base_node_inters[nodes[0]]]
        for L1 in L1s:
            if check_triangle(l[remainder],L1,L_R):
                full_inter_tuples.append(tuple([L1]))
    elif rank == 4:
        L1L2_prod = [i for i in itertools.product(base_node_inters[nodes[0]],base_node_inters[nodes[1]])]
        for L1L2 in L1L2_prod:
            L1,L2 = L1L2
            if check_triangle(L1,L2,L_R):
                good_tuple = (L1,L2)
                full_inter_tuples.append(good_tuple)
    elif rank == 5:
        L1L2_prod = [i for i in itertools.product(base_node_inters[nodes[0]],base_node_inters[nodes[1]])]
        next_node_inters = [get_intermediates_w(L1L2[0],L1L2[1]) for L1L2 in L1L2_prod]
        for L1L2,L3l in zip (L1L2_prod,next_node_inters):
            L1L2L3s = list(itertools.product([L1L2],L3l))
            for L1L2L3 in L1L2L3s:
                L1L2,L3 = L1L2L3
                L1,L2 = L1L2
                if check_triangle(l[remainder],L3,L_R):
                    good_tuple = (L1,L2,L3)
                    full_inter_tuples.append(good_tuple)
    elif rank == 6:
        L1L2L3_prod = [i for i in itertools.product(base_node_inters[nodes[0]],base_node_inters[nodes[1]],base_node_inters[nodes[2]])]
        next_node_inters = [get_intermediates_w(L1L2L3[0],L1L2L3[1]) for L1L2L3 in L1L2L3_prod]
        for L1L2L3 , L4l in zip (L1L2L3_prod,next_node_inters):
            L1L2L3L4s = list(itertools.product([L1L2L3] , L4l))
            for L1L2L3L4 in L1L2L3L4s:
                L1L2L3 , L4 = L1L2L3L4
                L1 , L2 , L3 = L1L2L3
                if check_triangle(L3 , L4 , L_R):
                    good_tuple = (L1 , L2 , L3 , L4)
                    full_inter_tuples.append(good_tuple)
    elif rank == 7:
        L1L2L3_prod = [i for i in itertools.product(base_node_inters[nodes[0]],base_node_inters[nodes[1]],base_node_inters[nodes[2]])]
        next_node_inters_l = [get_intermediates_w(L1L2L3[0],L1L2L3[1]) for L1L2L3 in L1L2L3_prod] #left hand branch 
        next_node_inters_r = [get_intermediates_w(L1L2L3[2],l[remainder]) for L1L2L3 in L1L2L3_prod] #right hand branch
        next_node_inters = [(L4,L5) for L4,L5 in zip(next_node_inters_l, next_node_inters_r)]
        for L1L2L3 , L45 in zip (L1L2L3_prod,next_node_inters):
            L1L2L3L4L5s = list(itertools.product([L1L2L3] , L45))
            for L1L2L3L4L5 in L1L2L3L4L5s:
                L1L2L3l , L45l = L1L2L3L4L5
                L1 , L2 , L3 = L1L2L3l
                L4 , L5 = L45l
                if check_triangle(L4 , L5 , L_R):
                    good_tuple = (L1 , L2 , L3 , L4, L5)
                    full_inter_tuples.append(good_tuple)
        
    elif rank == 8:
        L1L2L3L4_prod = [i for i in itertools.product(base_node_inters[nodes[0]],base_node_inters[nodes[1]],base_node_inters[nodes[2]],base_node_inters[nodes[3]])]
        next_node_inters_l = [get_intermediates_w(L1L2L3L4[0],L1L2L3L4[1]) for L1L2L3L4 in L1L2L3L4_prod] #left hand branch 
        next_node_inters_r = [get_intermediates_w(L1L2L3L4[2],L1L2L3L4[3]) for L1L2L3L4 in L1L2L3L4_prod] #right hand branch
        next_node_inters = [(L4,L5) for L4,L5 in zip(next_node_inters_l, next_node_inters_r)]
        for L1L2L3L4 , L56 in zip (L1L2L3L4_prod,next_node_inters):
            L1L2L3L4L5L6s = list(itertools.product([L1L2L3L4] , L56))
            for L1L2L3L4L5L6 in L1L2L3L4L5L6s:
                L1L2L3L4l , L56l = L1L2L3L4L5L6
                L1 , L2 , L3, L4 = L1L2L3L4l
                L5 , L6 = L56l
                if check_triangle(L5 , L6 , L_R):
                    good_tuple = (L1 , L2 , L3 , L4, L5, L6)
                    full_inter_tuples.append(good_tuple)
    else:
        raise ValueError("rank %d not implemented" % rank)

    return full_inter_tuples

def generate_l_LR(lrng , rank , L_R = 0 , M_R = 0):


    if M_R==0:
        even_flag = True
    elif M_R !=0:
        even_flag = False
    lmax = max(lrng)
    ls = []

    llst = ['%d'] * rank
    lstr = ','.join(b for b in llst)

    if rank == 1:
        ls.append('%d' % L_R)

    elif rank >1:
        all_ls = [b for b in itertools.product(lrng , repeat = rank)]
        if rank == 2:
            for ltup in all_ls:
                flag = check_triangle(ltup[0] , ltup[1] , L_R) and np.sum(ltup + (L_R , )) % 2 ==0
                if flag:
                    ls.append(lstr % ltup)
        elif rank == 3:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup) , L_R = L_R)
                by_node = vec_nodes(ltup , nodes , remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0] , node[1] , inter) for node,inter in zip(by_node,inters_i)]
                    inter_flags = [check_triangle(inters_i[0] , ltup[remainder] , L_R)]
                    parity_leaf_flag = np.sum([inters_i[0] , ltup[remainder] , L_R]) % 2 ==0
                    flags = li_flags + inter_flags
                    parity_all = np.sum(ltup) % 2 ==0
                    if all (flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)
        elif rank == 4:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup),L_R=L_R)
                by_node = vec_nodes(ltup,nodes,remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0] , node[1] , inter) for node , inter in zip(by_node , inters_i)]
                    inter_flags = [check_triangle(inters_i[0] , inters_i[1] , L_R)]
                    flags = li_flags + inter_flags
                    parity_all = np.sum(ltup) % 2 ==0
                    if all (flags) and parity_all:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 5:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup) , L_R = L_R)
                by_node = vec_nodes(ltup,nodes,remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0] , node[1] , inter) for node , inter in zip(by_node,inters_i)]
                    inter_flags = [check_triangle(inters_i[0] , inters_i[1] , inters_i[2]) , check_triangle(inters_i[2],ltup[remainder], L_R ) ]
                    flags = li_flags + inter_flags
                    if all (flags) and np.sum(ltup) % 2 == 0:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 6:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup) , L_R = L_R)
                by_node = vec_nodes(ltup , nodes , remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0],node[1],inter) for node,inter in zip(by_node,inters_i)]
                    inter_flags = [check_triangle(inters_i[0],inters_i[1],inters_i[3]), check_triangle(inters_i[2],inters_i[3], L_R ) ]
                    flags = li_flags + inter_flags
                    if all (flags) and np.sum(ltup) % 2 ==0:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 7:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup) , L_R = L_R)
                by_node = vec_nodes(ltup , nodes , remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0],node[1],inter) for node,inter in zip(by_node,inters_i)]
                    inter_flags = [check_triangle(inters_i[0],inters_i[1],inters_i[3]), check_triangle(inters_i[2],ltup[remainder], inters_i[4] ), check_triangle(inters[3],inters[4],L_R) ]
                    flags = li_flags + inter_flags
                    if all (flags) and np.sum(ltup) % 2 ==0:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)

        elif rank == 8:
            nodes,remainder = tree(list(all_ls[0]))
            for ltup in all_ls:
                inters = tree_l_inters(list(ltup) , L_R = L_R)
                by_node = vec_nodes(ltup , nodes , remainder)
                for inters_i in inters:
                    li_flags = [check_triangle(node[0],node[1],inter) for node,inter in zip(by_node,inters_i)]
                    inter_flags = [check_triangle(inters_i[0],inters_i[1],inters_i[4]), check_triangle(inters_i[2], inters_i[3], inters_i[5] ), check_triangle(inters[4],inters[5],L_R) ]
                    flags = li_flags + inter_flags
                    if all (flags) and np.sum(ltup) % 2 ==0:
                        lsub = lstr % ltup
                        if lsub not in ls:
                            ls.append(lsub)
                            
    return ls

        
def Reverse(lst):
    lst.reverse()
    return lst

def collect_by_l(nls):
    per_ltup = {}
    for nl in nls:
        mu0,mu,n,l = get_mu_n_l(nl)
        try:
            per_ltup[tuple([tuple(sorted(n)),tuple(sorted(l))])].append(nl)
        except KeyError:
            per_ltup[tuple([tuple(sorted(n)),tuple(sorted(l))])] = [nl]
    return flatten( list(per_ltup.values())), per_ltup

def generate_nl(rank,nmax,lmax,mumax=1,lmin=0,L_R=0,M_R=0,all_perms=False):
    # rank: int  - basis function rank to evaluate nl combinations for
    # nmax: int  - maximum value of the n quantum numbers in the nl vectors
    # lmax: int  - maximum value of the l quantum numbers in the nl vectors
    # mumax: int  - maximum value of the chemical variable in the munl vectors (default is none for single component system)
    # RETURN: list of munl vectors in string format mu0_mu1,mu2,...muk,n1,n2,..n_k,l1,l2,..l_k_L1-L2...-LK 
    # NOTE: All valid intermediates L are generated

    munl=[]

    murng = range(mumax)
    nrng = range(1,nmax+1)
    lrng = range(lmin,lmax+1)

    mus = ind_vec(murng,rank)
    ns = ind_vec(nrng,rank)
    ls = generate_l_LR(lrng,rank,L_R)

    linters_per_l = {l: tree_l_inters([int(b) for b in l.split(',')] , L_R = 0) for l in ls }
    

    munllst = ['%d']*int(rank*3)
    munlstr = ','.join(b for b in munllst)
    for mu0 in murng:
        for cmbo in itertools.product(mus,ns,ls):
            mu,n,l = cmbo
                
            linters = linters_per_l[l]
            musplt = [int(k) for k in mu.split(',')]
            nsplt = [int(k) for k in n.split(',')]
            lsplt = [int(k) for k in l.split(',')]
            x = [(musplt[i],lsplt[i],nsplt[i]) for i in range(rank)]
            srt = sorted(x)
            if not all_perms:
                conds = x==srt
            elif all_perms:
                conds = tuple(lsplt) == tuple(sorted(lsplt))
            if conds:
                stmp = '%d_' % mu0 +  munlstr  % tuple( musplt+nsplt+lsplt)
                #if stmp not in munl:
                for linter in linters:
                    linter_str_lst = ['%d']*len(linter)
                    linter_str = '-'.join(b for b in linter_str_lst) % linter
                    munlL = stmp + '_' + linter_str
                    munl.append(munlL)
    munl = list(set(munl))
    return munl

def generate_nl_noL(rank,nmax,lmax,mumax=1,L_R=0,M_R=0):
    # rank: int  - basis function rank to evaluate nl combinations for
    # nmax: int  - maximum value of the n quantum numbers in the nl vectors
    # lmax: int  - maximum value of the l quantum numbers in the nl vectors
    # mumax: int  - maximum value of the chemical variable in the munl vectors (default is none for single component system)
    # RETURN: list of munl vectors in string format mu0_mu1,mu2,...muk,n1,n2,..n_k,l1,l2,..l_k
    # NOTE: Intermediates L are not generated

    munl=[]

    murng = range(mumax)
    nrng = range(1,nmax+1)
    lrng = range(lmax+1)

    mus = ind_vec(murng,rank)
    ns = ind_vec(nrng,rank)
    lsi = generate_l_LR(lrng,rank,L_R)
    ls = []
    for lstr in lsi:
        li=[int(b) for b in lstr.split(',')]
        li=sorted(li)
        ls.append(li)
    ls = sorted(list(set(ls)))
    linters_per_l = {l: tree_l_inters(l , L_R = 0) for l in ls }
    

    munllst = ['%d']*int(rank*3)
    munlstr = ','.join(b for b in munllst)
    for mu0 in murng:
        for cmbo in itertools.product(mus,ns,ls):
            mu,n,l = cmbo
                
            linters = linters_per_l[l]
            musplt = [int(k) for k in mu.split(',')]
            nsplt = [int(k) for k in n.split(',')]
            lsplt = [int(k) for k in l.split(',')]
            x = [(musplt[i],lsplt[i],nsplt[i]) for i in range(rank)]
            srt = sorted(x)
            if x == srt:
                stmp = '%d_' % mu0 +  munlstr  % tuple( musplt+nsplt+lsplt)
                if stmp not in munl:
                    munl.append(stmp)
    return munl
