from fitsnap3lib.lib.sym_ACE.lib.coupling_tree import * 

#logical for printing extra info
verbose = False

#global dicts for various symmetries
global_lsyms = {}
global_lsemi = {}
global_parts = {}
global_orbits = {}


def permutation_adapted_lL(l,semistandardflag=True):
    # This is a function to return the permutation-adapted angular 
    #  basis labels. It finds the varsigma(l) permutations for a 
    #  provided block of l_i (as a list)
    
    #----------------------------------------------------------------
    # Define the identity l_o based on degeneracy and the coupling 
    # scheme. (This is no longer necessary but it makes for smaller
    # amounts of permutations to sift over to find the full set of
    #  { \varsigma(l) }
    #----------------------------------------------------------------
    N = len(l)
    uniques = list(set(l))
    tmp = l.copy()
    tmp.sort(key=Counter(l).get,reverse=True)
    uniques.sort(key=Counter(tmp).get,reverse=True)
    count_uniques =[l.count(u) for u in uniques]
    mp = {uniques[i]:i for i in range(len(uniques))}
    mprev = {i:uniques[i] for i in range(len(uniques))}
    mapped = [mp[t] for t in tmp]
    
    if len(uniques) == math.floor(len(l)/2):# and 0 in uniques:
        l.sort(key=Counter(l).get,reverse=True)
        mapped.sort(key=Counter(mapped).get,reverse=True)
    else:
        l = sort_pair(l)
        mapped = sort_pair(mapped)
    assert len(l) <= 6, "symmetry reduction is only needed for rank 4 + descriptors. Use a rank between 4 and 6, automorphism groups for rank >= 7 to be added soon."
    N = len(l)
    if verbose:
        print ('lrep',l)
    #----------------------------------------------------------------
    # End definition of the identity
    #----------------------------------------------------------------


    #----------------------------------------------------------------
    # Use young subgroup fillings to reduce a full search over S_N
    #  to obtain \varsigma(l) and irrep. The young subgroup used is
    #  S_2 \otimes S_2 \otimes ... S_2  - for N/2 iterations if N is
    #  even and
    #  S_2 \otimes S_2 \otimes ... \otimes S_1 - for floor(N/2) 
    #  iterations if N is odd.
    #----------------------------------------------------------------
    
    try:
        sigma_c_parts = global_parts[N]
    except KeyError:
        
        ysgi = Young_Subgroup(N)
        sigma_c_parts = ysgi.sigma_c_partitions(max_orbit=N)
        sigma_c_parts.sort(key=lambda x: x.count(2),reverse=True)
        sigma_c_parts.sort(key=lambda x: tuple([i%2==0 for i in x]),reverse=True)
        sigma_c_parts.sort(key=lambda x: max(x),reverse=True)
        global_parts[N] = sigma_c_parts
        if verbose:
            print (sigma_c_parts)
    try:
        lperms = global_lsyms[tuple(mapped)]
        part_per_fill = global_orbits[tuple(mapped)]
        lperms_semistandard = global_lsemi[tuple(mapped)]
    except KeyError:
        ysgi = Young_Subgroup(N)
        ysgi.subgroup_fill(mapped,sigma_c_parts,sigma_c_symmetric=True,semistandard=False)
        lperms = ysgi.fills.copy()
        lperms = ysgi.reduce_list(mapped,lperms)
        part_per_fill = ysgi.partitions_per_fill
        ysgi.subgroup_fill(mapped,sigma_c_parts,sigma_c_symmetric=True,semistandard=True)
        lperms_semistandard = ysgi.fills.copy()
        standard_part_per_fill = ysgi.partitions_per_fill
        global_lsyms[tuple(mapped)] = lperms
        global_orbits[tuple(mapped)] = part_per_fill
        global_lsemi[tuple(mapped)] = lperms_semistandard
        if verbose:
            print ('lperms_ysg_not',lperms)
            print ('lperms_ysg_semistandard',lperms_semistandard)
            print ('highest symmetry orbits per filling', part_per_fill)
    myvarsigma_l = [tuple([mprev[k] for k in lp]) for lp in lperms]
    myvarsigma_l = sorted(myvarsigma_l)
    varsigma_l = []

    L_per_varsigma = {}

    for vsl in myvarsigma_l:
        #print ('vsl in loop',vsl)
        varsigma_l.append(vsl)
        L_lp = tree_l_inters(vsl)
        nodes,remainder = tree(vsl)
        lnodes_lp = group_vec_by_node(vsl,nodes,remainder)
        L_lp_filtered = parity_filter(lnodes_lp,L_lp)
        L_per_varsigma[ vsl]=L_lp_filtered

    reps_per_varsigma = {}
    for fill,irreps in part_per_fill.items():
        lmp = tuple([mprev[k] for k in fill])
        reps_per_varsigma[lmp] = irreps
    
    # the following loop eliminates dependent labels due to
    #  degeneracies in internal nodes. (e.g. removes trees with
    #  equivalent branch structures once intermediates are applied)
    quick_build = []
    orbids = []
    leafids = []
    used_l = []
    all_autos = []
    ysg = Young_Subgroup(N)
    for lp, Lp_lst in L_per_varsigma.items():
        ti = Tree_ID(lp,Lp_lst[-1])
        orbi = ti.return_orbit_l_ID()
        leafi = ti.return_leaf_l_ID()
        ysg.set_inds(lp)
        automorphisms = get_auto_part(tuple(lp),tuple([len(lp)]),add_degen_autos=False,part_only=False)
        conj1, applied_conj,conj_list = ysg.apply_automorphism_conjugation(my_automorphisms=automorphisms)
        # first catch any binary trees that share the same
        #  leaf structure as the coupling permutation, \sigma_c
        if leafi not in leafids:
            quick_build.append( (lp,Lp_lst[-1]))
            orbids.append(orbi)
            leafids.append(leafi)
            all_autos.extend(conj_list)
        # secondly, catch binary trees that share more symmetry
        #  even if it has a leaf structure compatible with \sigma_c
        #  as well. (Needed for degenerate l_i)
        if orbi not in orbids:
            quick_build.append( (lp,Lp_lst[-1]))
            orbids.append(orbi)
            leafids.append(leafi)
            all_autos.extend(conj_list)

    varsigma_lL = quick_build.copy()
    # for each varsigma(l) - provide the highest symmetry partition
    #  (isomorphic to an irrep in S_N) it is compatible with
    highest_sym_reps = {}
    for lL in varsigma_lL:
        varsigmali,intersi = lL
        reps = reps_per_varsigma[varsigmali]
        highest_sym_reps[lL] = reps[0]
            
    return varsigma_lL, highest_sym_reps

def permutation_adapted_nlL(n,l,semistandardflag=True):
    # flag to return in descriptor label format for FitSNAP
    return_desclabels = True

    #------------------------------------------------------------------
    # generate PA-RI basis and corresponding irreps of SN
    lLs, SN_irreps = permutation_adapted_lL(l,semistandardflag)

    # initialize lists for permutation adapted nl labels
    #  and used binary trees, respectively
    varsigma_nls = []
    used_nl_reps = []

    # make all permutations of n ( the function below removes explicit
    #  redundancies such as n_1=(1111), n_2=(1111) that are generated
    #  when using python-native enumerations (e.g. itertools.permutations)
    ns = unique_perms(n)
    
    # using the set { \varsigma_l } and the corresponding highest symmetry
    #  set of orbits used to generate \varsigma_l, find all unique nl
    #  nl trees. (To go from permutation-adapted angular indices to 
    #  permutation-adapted radial + angular indices, we add more leaves to the
    #  tree.
    for lL,irrep in SN_irreps.items():
        li,Li = lL
        ti = Tree_ID(li,Li)
        loid = ti.return_orbit_l_ID(orbit=irrep)
        for nperm in ns:
            nloid = ti.return_orbit_nl_ID(nperm,orbit=irrep)
            if verbose:
                print (nloid)
            if nloid not in used_nl_reps:
                used_nl_reps.append(nloid)
                varsigma_nls.append((nperm,li,Li))

    

    #------------------------------------------------------------------

    # Remaining code in this function is to return descriptor labels in 
    #  fitsnap format.
    descriptor_labels = []
    # enforce single element in lite version
    nelements=1
    N = len(l)
    for mu0 in range(nelements):
        for munlL in varsigma_nls:
            munlL = tuple([ tuple([0]*N)  ]) + munlL
            st='%d_' % mu0
            tmp=  ','.join([b for b in ['%d']*N*(3)])
            tmp = tmp % tuple(flatten(munlL[:-1]))
            st +=tmp
            st+=  '_'
            st+= '-'.join(str(b) for b in munlL[3])
            descriptor_labels.append(st)
    if return_desclabels:
        return descriptor_labels
    else:
        return varsigma_nls


def permutation_adapted_munlL(mu,n,l,nelements=1,semistandardflag=True):
    # flag to return in descriptor label format for FitSNAP
    return_desclabels = True

    #------------------------------------------------------------------
    # generate PA-RI basis and corresponding irreps of SN
    lLs, SN_irreps = permutation_adapted_lL(l,semistandardflag)

    # initialize lists for permutation adapted nl labels
    #  and used binary trees, respectively
    varsigma_nls = []
    used_nl_reps = []

    # make all permutations of n ( the function below removes explicit
    #  redundancies such as n_1=(1111), n_2=(1111) that are generated
    #  when using python-native enumerations (e.g. itertools.permutations)
    ns = unique_perms(n)
    mus = unique_perms(mu)
    
    # using the set { \varsigma_l } and the corresponding highest symmetry
    #  set of orbits used to generate \varsigma_l, find all unique nl
    #  nl trees. (To go from permutation-adapted angular indices to 
    #  permutation-adapted radial + angular indices, we add more leaves to the
    #  tree.
    for lL,irrep in SN_irreps.items():
        li,Li = lL
        ti = Tree_ID(li,Li)
        loid = ti.return_orbit_l_ID(orbit=irrep)
        for nperm in ns:
            for muperm in mus:
                nloid = ti.return_orbit_munl_ID(muperm,nperm,orbit=irrep)
                if verbose:
                    print (nloid)
                if nloid not in used_nl_reps:
                    used_nl_reps.append(nloid)
                    varsigma_nls.append((muperm,nperm,li,Li))

    

    #------------------------------------------------------------------

    # Remaining code in this function is to return descriptor labels in 
    #  fitsnap format.
    descriptor_labels = []
    N = len(l)
    for mu0 in range(nelements):
        for munlL in varsigma_nls:
            munlL = munlL
            st='%d_' % mu0
            tmp=  ','.join([b for b in ['%d']*N*(3)])
            tmp = tmp % tuple(flatten(munlL[:-1]))
            st +=tmp
            st +=  '_'
            st += '-'.join(str(b) for b in munlL[3])
            descriptor_labels.append(st)
    if return_desclabels:
        return descriptor_labels
    else:
        return varsigma_nls



def descriptor_labels_YSG(rank,nmax,lmax,mumax=1,lmin=1):
    if mumax ==0:
        mumax = 1
        
    if rank >= 4:
        lrng = list(range(lmin,lmax+1))
        nrng = list(range(1,nmax+1))
        murng = list(range(mumax))
        # function to return all l vectors obeying angular momentum
        #  N-dimensional polygon conditions for coupling scheme in
        #  paper.
        lstrs = generate_l_LR(lrng , rank , L_R = 0)

        ns = [i for i in itertools.combinations_with_replacement(nrng,rank)]
        mus = [i for i in itertools.combinations_with_replacement(murng,rank)]
        used_ls = []
        labels = []
        
        # loop over n and l vectors and build permutation-adapted blocks
        #  for all combinations of these vectors
        if mumax ==0 or mumax==1:
            for lstr in sorted(list(set(lstrs))):
                l = [int(k) for k in lstr.split(',')]
                N = len(l)
                for n in ns:
                    lrep = tuple([ tuple(sorted(l)), tuple(sorted(n))])
                    if lrep not in used_ls:
                        nls = permutation_adapted_nlL(n,l)
                        used_ls.append(lrep)
                        labels.extend(nls)
        else:
            for lstr in sorted(list(set(lstrs))):
                l = [int(k) for k in lstr.split(',')]
                N = len(l)
                for n in ns:
                    for mu in mus:
                        lrep = tuple([ tuple(sorted(l)), tuple(sorted(n)), tuple(sorted(mu))])
                        if lrep not in used_ls:
                            munls = permutation_adapted_munlL(mu,n,l,nelements=mumax)
                            used_ls.append(lrep)
                            labels.extend(munls)
            

    elif rank < 4:
        # no symmetry reduction required for rank <= 3
        # use typical lexicographical ordering for such cases 
        labels = generate_nl(rank,nmax,lmax,mumax=mumax,lmin=lmin,L_R=0,M_R=0,all_perms=False)  
    munltups = [get_mu_n_l(nu,return_L=True) for nu in labels]
    return labels
