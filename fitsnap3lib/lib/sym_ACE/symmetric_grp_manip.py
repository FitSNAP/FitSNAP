import numpy as np
from sympy.combinatorics import Permutation
from fitsnap3lib.lib.sym_ACE.tree_sorting import *
import itertools

def leaf_filter(lperms):
    rank = len(lperms[0])
    part = local_sigma_c_partitions[rank][-1]
    filtered = []
    if rank <=5 :
        for lperm in lperms:
            grouped = group_vec_by_orbits(lperm,part)
            subgroup = [g for g in grouped if len(g) > 1]
            if tuple(sorted(subgroup)) == tuple(subgroup):
                filtered.append(lperm)
            else:
                pass
    elif rank in [6,7]:
        for lperm in lperms:
            grouped = group_vec_by_orbits(lperm,part)
            subgroup = [g for g in grouped[:2]]
            if tuple(sorted(subgroup)) == tuple(subgroup):
                filtered.append(lperm)
            else:
                pass
    else:
        raise ValueError('manual orbit construction for rank %d not implemented yet' % rank)
        
    return filtered

def find_degenerate_indices(lst):
    degenerate_indices = {}
    for i in range(len(lst)):
        if lst.count(lst[i]) >= 1:
        #if lst.count(lst[i]) > 1:
            if lst[i] not in degenerate_indices:
                degenerate_indices[lst[i]] = []
            degenerate_indices[lst[i]].append(i)
    return degenerate_indices

def check_sequential(lst):
    flags = []
    if len(lst) > 1:
        for i in range(len(lst) - 1):
            flags.append(lst[i] + 1 == lst[i+1])
        return all(flags)
    else:
        return True

def find_sequential_indices(lst):
    sequential_indices = []
    for i in range(len(lst) - 1):
        if lst[i] + 1 == lst[i+1]:
            sequential_indices.extend([i, i+1])
    return sequential_indices

def get_degen_orb(l):
    rank = len(l)
    degen_ind_dict = find_degenerate_indices(l)
    partition = []
    #print ('degen_dct',degen_ind_dict)
    inds_per_orbit = {}
    for degenval, matching_inds in degen_ind_dict.items():
        sequential_inds = sorted(list(set(find_sequential_indices(matching_inds))))
        #print ('seq',sequential_inds,'match',matching_inds)
        this_orbit = tuple(matching_inds)
        partition.append(this_orbit)
        try:
            inds_per_orbit[degenval].extend(matching_inds)
        except KeyError:
            inds_per_orbit[degenval] = []
            inds_per_orbit[degenval].extend(matching_inds)
    partition = tuple(partition)    
    part_tup = tuple([len(ki) for ki in partition])
    return part_tup,partition

def get_young_map(partition_inds):
    current_size_tup = tuple([len(ki) for ki in partition_inds])
    sorted_size_tup = tuple(sorted(current_size_tup))
    full = flatten(partition_inds)
    full = tuple(sorted(full))
    #for iorb, orb in enumerate(partition_inds):

def enforce_sorted_orbit(partition_inds):
    rank = len(flatten(partition_inds))
    couple_ref = group_vec_by_orbits(list(range(rank)),local_sigma_c_partitions[rank][-1])
    new_partition = []
    flag = all([check_sequential(oi) and oi in couple_ref for oi in partition_inds])
    if not flag:
        flags = [check_sequential(oi) == oi and oi in couple_ref for oi in partition_inds]
        for iflag, orbit_flag in enumerate(flags):
            if not orbit_flag:
                symmetric_sub_orbits = []
                for oind,couple_orb in enumerate(couple_ref):
                    has_symmetric_sub_orbit = all([oo in partition_inds[iflag] for oo in couple_orb])
                    if has_symmetric_sub_orbit and couple_orb not in symmetric_sub_orbits:
                        symmetric_sub_orbits.append(couple_orb)
                if len(symmetric_sub_orbits) == 0:
                    new_orbit = [tuple([ki]) for ki in flatten(partition_inds[iflag])]
                else:
                    remain = [ tuple([ki]) for ki in flatten(partition_inds[iflag]) if ki not in flatten(symmetric_sub_orbits)]
                    new_orbit = symmetric_sub_orbits + remain
                new_partition.extend(new_orbit)
                new_partition = sorted(new_partition)
            else:
                new_partition.extend(partition_inds[iflag])
    else:
        new_partition = tuple(list(partition_inds).copy())
    part_tup = tuple([len(ki) for ki in new_partition])
    return part_tup,tuple(new_partition)

def get_sequential_degen_orb(l):
    degen_ind_dict = find_degenerate_indices(l)
    partition = []
    for degenval, matching_inds in degen_ind_dict.items():
        sequential_inds = sorted(list(set(find_sequential_indices(matching_inds))))
        if len(sequential_inds) != 0:
            this_orbit = [tuple(sequential_inds)]
        else:
            this_orbit = [(0,)]
        #this_orbit = [tuple(matching_inds)]
        if len(flatten(this_orbit)) == len(matching_inds):
            this_orbit = this_orbit
        elif len(flatten(this_orbit)) != len(matching_inds):
            #remaining = matching_inds[len(flatten(this_orbit))-1:]
            remaining = matching_inds[len(flatten(this_orbit)):]
            for ii in remaining:
                this_orbit.append((ii,))

        partition.extend(this_orbit)
    if len(flatten(partition)) != len(l):
        remaining_inds = [ri for ri in range(len(l)) if ri not in flatten(partition)]
        partition.extend([(ri,) for ri in remaining_inds])
    #partition = sorted(partition)

    return tuple([len(k) for k in partition]), partition
