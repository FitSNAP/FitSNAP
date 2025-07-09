from fitsnap3lib.lib.sym_ACE.gen_labels import *

def simple_parity_filt(l, inters, L_R, even = True):
    nodes,remainder = tree(l)
    base_ls = group_vec_by_node(l,nodes,remainder=remainder)
    base_ls = [list(k) for k in base_ls]
    if even:
        assert (np.sum(l) % 2) == 0, "must have sum{l_i} = even for even parity definition"
        if len(l) == 4:
            inters_filt = [i for i in inters if  np.sum([i[0]] + base_ls[0])  % 2 == 0 and np.sum([i[1]] + base_ls[1]) % 2 == 0]
        else:
            if remainder == None:
                inters_filt = [i for i in inters if  all([ np.sum([i[ind]] + base_ls[ind]) %2 ==0  for ind in range(len(base_ls))   ]) ] 
            else:
                inters_filt = [i for i in inters if  all([ np.sum([i[ind]] + base_ls[ind]) %2 ==0  for ind in range(len(base_ls))   ]) ] 
            
    else:
        assert (np.sum(l) % 2) !=0, "must have sum{l_i} = odd for odd parity definition"
        print ('WARNING! You are using an odd parity tree. Check your labels to make sure this is what you want (this is for fitting vector quantities!)')
        if len(l) == 4:
            inters_filt = [inters[ind] for ind,i in enumerate(base_ls) if  np.sum( [inters[ind][i]] + list(i)  )   % 2   != 0 ]
    return inters_filt

def max_inters(l):
    nodes,remainder = tree(l)
    base_ls = group_vec_by_node(l,nodes,remainder=remainder)
    if remainder == None:
        base_Ls = [ get_intermediates(ln) for ln in base_ls]
    else:
        base_Ls = [ get_intermediates(ln) for ln in base_ls[:-1]]
    max_base_Ls = [max(L) for L in base_Ls]
    if len(l) == 4:
        max_LR = max(get_intermediates(max_base_Ls))
        max_Ls = tuple(max_base_Ls)
    elif len(l) == 5:
        max_L3 = max(get_intermediates(max_base_Ls))
        max_LR = max(get_intermediates([max_L3,l[remainder]]))
        max_Ls = tuple(max_base_Ls + [max_L3])
    elif len(l) == 6:
        max_L4= max(get_intermediates(max_base_Ls[:2]))
        max_L3 = max_base_Ls[2]
        max_LR = max(get_intermediates([max_L3,max_L4]))
        max_Ls = tuple(max_base_Ls[:2] + [max_L4])
    #print ('max LR',max_LR)
    return max_Ls, max_LR

def min_inters(l):
    nodes,remainder = tree(l)
    base_ls = group_vec_by_node(l,nodes,remainder=remainder)
    if remainder == None:
        base_Ls = [ get_intermediates(ln) for ln in base_ls]
    else:
        base_Ls = [ get_intermediates(ln) for ln in base_ls[:-1]]
    min_base_Ls = [min(L) for L in base_Ls]
    #print (max_base_Ls)
    if len(l) == 4:
        min_LR = min(get_intermediates(min_base_Ls))
        min_Ls = tuple(min_base_Ls)
    elif len(l) == 5:
        min_L3 = min(get_intermediates(min_base_Ls))
        min_LR = min(get_intermediates([min_L3,l[remainder]]))
        min_Ls = tuple(min_base_Ls + [min_L3])
    elif len(l) == 6:
        min_L4= min(get_intermediates(min_base_Ls[:2]))
        min_L3 = min_base_Ls[2]
        min_LR = min(get_intermediates([min_L3,min_L4]))
        min_Ls = tuple(min_base_Ls[:2] + [min_L4])
    return min_Ls, min_LR

def LR_set(l,LR_max,LR_min=0):
    if np.sum(l) % 2 ==0:
        even = True
    elif np.sum(l) %2 != 0:
        even = False
    if even:
        LRs = [i for i in range(LR_min,LR_max+1) if i % 2 ==0]
    else: 
        LRs = [i for i in range(LR_min,LR_max+1) if i % 2 !=0]
    
    return LRs,even

def inters_per_LR(l,LRset,even):
    spanning_dict = {LR: [] for LR in LRset }

    for LR in LRset:
        unf_inters = tree_l_inters(l,LR)
        f_inters = simple_parity_filt(l, unf_inters, LR,even)
        spanning_dict[LR].extend(f_inters)
    return spanning_dict
