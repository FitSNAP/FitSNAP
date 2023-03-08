from sympy.combinatorics import Permutation
from fitsnap3lib.lib.sym_ACE.lib.sylow_lib import *
import math
from fitsnap3lib.lib.sym_ACE.gen_labels import *

#library of functions and classes for building young diagrams and filling them
# this includes young subgroup fillings within orbits of irreps of S_N
global_sigma_c_parts = {}
def complimenting_conjugation(h1_trans,l,cs):
    #needs to return:
    # (1,3) -> (0,2)
    # (0,5) -> (1,4)
    # (0,3) -> (1,2) ...
    symmetric_compliments = []
    h1_index = { i : [c for c in cs if i in c][0] for i in h1_trans }
    compliment = [[newidx for newidx in h1_index[idx] if newidx != idx][0] for idx in h1_trans]
    return tuple(compliment)

def parity_filter(lnodes,linters, L_R=0):
    # enforces parity constraints on a lL combination
    remainder = None
    l = flatten(lnodes)
    if any([len(lnode) ==1 for lnode in lnodes]):
        remainder = lnodes[-1][0]
        lnodes = lnodes[:-1]

    rank = len(l)
    all_flags = []
    if rank == 4:
        for linter in linters:
            flags = [(lnode[0] + lnode[1] + linteri) % 2 == 0 for lnode,linteri in zip(lnodes,linter)]
            all_flags.append(all(flags))
    elif rank == 5:
        for linter in linters:
            flags = [(lnode[0] + lnode[1] + linteri) % 2 == 0 for lnode,linteri in zip(lnodes,linter)]
            if remainder != None:
                flags.append((remainder + linter[0] + linter[1] ) % 2 ==0)
                flags.append((remainder + linter[2] + L_R ) % 2 ==0)
                flags.append((linter[0] + linter[1] + linter[2]) % 2 ==0)
            all_flags.append(all(flags))
    elif rank == 6:
        for linter in linters:
            flags = [(lnode[0] + lnode[1] + linteri) % 2 ==0 for lnode,linteri in zip(lnodes,linter)]
            flags.append((linter[0] + linter[1] + linter[2]) % 2 ==0)
            all_flags.append(all(flags))

    reduced_linters = [linter for ind,linter in enumerate(linters) if all_flags[ind]]
    return reduced_linters

def filled_perm(tups,rank):
    allinds = list(range(rank))
    try:
        remainders = [ i for i in allinds if i not in flatten(tups)]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    except TypeError:
        remainders = [ i for i in allinds if i not in flatten(flatten(tups))]
        alltups = tups + tuple([tuple([k]) for k in remainders])
    return(Permutation(alltups))

def is_column_sort(partitionfill,strict_col_sort = False):
    lens = [len(x) for x in partitionfill]
    ranges = [list(range(ln)) for ln in lens]
    cols = list(set(flatten(ranges)))
    bycol = {col:[] for col in cols}
    for subrange, orbitlst in zip(ranges,partitionfill):
        for colidx,orbitval in zip(subrange,orbitlst):
            bycol[colidx].append(orbitval)
    coltups = [tuple(bycol[colidx]) for colidx in cols]
    sortedcols = [tuple(sorted(bycol[colidx])) for colidx in cols]
    #check to see if columns are sorted
    sortedcol_flag = all([ a==b for a,b in zip(coltups,sortedcols)])
    if strict_col_sort:
        sortedcol_flag = sortedcol_flag and all([ len(list(set(a))) == len(a) for a in coltups])
    return sortedcol_flag

def is_row_sort(partitionfill):
    all_srt = []
    for orbit in partitionfill:
        logi = tuple(sorted(orbit)) == orbit
        all_srt.append(logi)
    return all(all_srt)

class Young_Subgroup:
    # Class for young tableau with subgroup filling options

    def __init__(self,
            rank):
        self.rank = rank
        self.partition = None
        self.fills = None

    def set_inds(self,l):
        self.inds = l

    def set_partition(self,partition):
        self.partition = partition

    def apply_transpose_conjugation(self,conj_per_part):
        #applies transposition conjugations (useful for within pairwise orbits)
        veccount = 0
        vecparts = []
        for part in self.partition:
            partl = []
            for i in range(veccount,veccount + part):
                partl.append(self.inds[i])
            veccount += part 
            vecparts.append(partl)

        conjed = [ ]
        for ipart,vecpart in enumerate(vecparts):
            if conj_per_part[ipart]:
                orbit_tmp = Reverse(vecpart)
            else:
                orbit_tmp = vecpart
            conjed.append(orbit_tmp)
        return flatten(conjed)
    
    def apply_automorphism_conjugation(self,exclude_ops=[],my_automorphisms=None):
        ipart = self.partition
        if my_automorphisms == None:
            my_automorphisms = get_auto_part(tuple(self.inds),tuple(ipart),add_degen_autos=True)
        inds_identity = tuple(self.inds)
        conjugation = tuple(self.inds)
        conjugations = []
        used_operation = tuple([ tuple([k]) for k in range(len(self.inds))])
        exclude_ops.append(used_operation)
        for auto in my_automorphisms:
            p = Permutation(auto)
            permed = p(inds_identity)
            permed = tuple(permed)
            if auto not in exclude_ops:
                conjugation = permed
                used_operation = auto
                break
        
        for auto in my_automorphisms:
            p = Permutation(auto)
            permed = p(inds_identity)
            permed = tuple(permed)
            conj = permed
            used_operation = auto
            if conj not in conjugations:
                conjugations.append(conj)
        return conjugation,used_operation,conjugations


    def sigma_c_partitions(self,max_orbit):
        #returns a list of partitions compatible with sigma_c
        nodes,remainder = tree(range(self.rank))
        self.nodes = nodes
        self.remainder = remainder
        max_nc2 = math.floor((max_orbit)/2)
        min_orbits = 1
        tst = max_nc2 % 2
        #min_orbits = max_nc2 #+= tst
        min_orbits  += tst
        tmp_max = math.ceil(self.rank/2)
        min_nc1 = 0
        if remainder != None:
            min_nc1 = 1
        if max_orbit != None:
            max_orbits = max_orbit
        elif max_orbit == None:
            max_orbits = max_nc2 + min_nc1
        orb_base = [i for i in range(1,self.rank+1) if i %2 ==0 or i == 1]

        if remainder != None:
            orb_base.append(1)
        try:
            good_partitions = global_sigma_c_parts[self.rank]
        except KeyError:
            good_partitions = []
            for norbits in range(min_orbits , max_orbit + 1):
                possible_parts = [tuple(Reverse(sorted(p))) for p in itertools.product(orb_base , repeat = norbits) if np.sum(p) == self.rank  and max(p) <= 2**max_nc2]
                good_partitions.extend(possible_parts)
            global_sigma_c_parts[self.rank] = list(set(good_partitions))
        return list(set(good_partitions))

    def check_subgroup_fill(self,partition,inds,sigma_c_symmetric = False , semistandard = True, allow_conjugations=False):
        tmpi = sorted(inds)
        unique_tmpi = list(set(tmpi))
        ii_to_inds = {}
        indi_to_ii = {}
        for ii,indi in zip(range(len(unique_tmpi)),unique_tmpi):
            ii_to_inds[ii] = indi
            indi_to_ii[indi] = ii

        place_holders = [indi_to_ii[ik] for ik in inds]
        if self.partition == None:
            self.partition = partition
        all_perms = [tuple(place_holders)]
        all_fills = [group_vec_by_orbits(perm,partition) for perm in all_perms]
        nodes,remainder = tree(inds)
        if not semistandard:
            if sigma_c_symmetric:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_row_sort(fill) and  is_row_sort( group_vec_by_node(flatten(fill),nodes,remainder) ) ]
            else:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_row_sort(fill)]
        elif semistandard:
            if sigma_c_symmetric:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_column_sort(fill) and  is_row_sort(fill) and  is_row_sort( group_vec_by_node(flatten(fill),nodes,remainder) ) ]
            else:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_column_sort(fill) and is_row_sort(fill)]
        actual_fills = []
        for fill in return_fills:
            actual_fills.append(tuple( [ ii_to_inds[ij] for ij in fill]  ) )
        return tuple(inds) in actual_fills, actual_fills

    def check_subgroup_fill_returnfill(self,partition,inds,sigma_c_symmetric = False , semistandard = True, allow_conjugations=False):
        tmpi = sorted(inds)
        unique_tmpi = list(set(tmpi))
        ii_to_inds = {}
        indi_to_ii = {}
        for ii,indi in zip(range(len(unique_tmpi)),unique_tmpi):
            ii_to_inds[ii] = indi
            indi_to_ii[indi] = ii

        place_holders = [indi_to_ii[ik] for ik in inds]

        if self.partition == None:
            self.partition = partition
        all_perms = unique_perms(place_holders)
        all_fills = [group_vec_by_orbits(perm,partition) for perm in all_perms]
        nodes,remainder = tree(inds)
        if not semistandard:
            if sigma_c_symmetric:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_row_sort(fill) and  is_row_sort( group_vec_by_node(flatten(fill),nodes,remainder) ) ]
            else:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_row_sort(fill)]
        elif semistandard:
            if sigma_c_symmetric:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_column_sort(fill) and  is_row_sort(fill) and  is_row_sort( group_vec_by_node(flatten(fill),nodes,remainder) ) ]
            else:
                return_fills = [tuple(flatten(fill)) for fill in all_fills if is_column_sort(fill) and is_row_sort(fill)]
        actual_fills = []
        for fill in return_fills:
            actual_fills.append(tuple( [ ii_to_inds[ij] for ij in fill]  ) )
        return tuple(inds) in actual_fills, actual_fills

    def reduce_list(self,inds,lst):
        partitions = self.sigma_c_partitions(len(inds) -1)
        G_N = get_auto_part(inds,partitions[0],add_degen_autos=True,part_only=False)
        subt = get_auto_part(inds,partitions[0],add_degen_autos=False,part_only=False,subtree=True)
        matched = {lid:[] for lid in lst}
        keep = []
        dont_keep = []
        for lid in lst:
            applied_perms = [tuple(Permutation(filled_perm(pi,len(inds)))(lid)) for pi in G_N]
            applied_permssub = [tuple(Permutation(filled_perm(pi,len(inds)))(lid)) for pi in subt]
            applied_permssub = [ik for ik in applied_permssub if ik != lid]
            not_thisid = [k for k in applied_perms if k != lid]
            these_matches = [k for k in not_thisid if k in lst]
            for mtch in these_matches:
                if mtch not in matched[lid]:
                    matched[lid].append(mtch)
            if lid not in dont_keep:
                keep.append(lid)
                dont_keep.extend(these_matches)
                dont_keep.append(lid)
        return list(set(keep))
    
    def subgroup_fill(self,inds,partitions=None,max_orbit=None,sigma_c_symmetric=False,semistandard=True,lreduce=False):
        if partitions == None:
            partitions = self.sigma_c_partitions(max_orbit)
        if len(set(inds)) == 1:
            partitions = [tuple([len(inds)])] +  partitions
        subgroup_fills = []
        fills_perpart = {tuple(partition):[] for partition in partitions}
        part_perfill = {}
        all_perms = unique_perms(inds)
        if len(inds) %2 ==0:
            perms_raw = [p for p in itertools.permutations(range(len(inds)))]
        elif len(inds) %2 != 0:
            # removes redundant permutations for rank 5
            perms_raw = [p for p in itertools.permutations(range(len(inds))) if p[-1] == range(len(inds))[-1] or p[-1] == range(len(inds))[0]]
        # get the full automorphism group including any expansion due to degeneracy
        G_N = get_auto_part(inds,partitions[0],add_degen_autos=True,part_only=False)
        applied_perms = [tuple(Permutation(filled_perm(pi,len(inds)))(inds)) for pi in G_N]
        # collect a group of permutations \sigma \in S_N \notin G_N 
        idi = [tuple([ki]) for ki in range(len(inds))]
        H_N = [tuple(idi) ]
        for raw_perm in perms_raw:
            P = Permutation(raw_perm)
            cyc = P.full_cyclic_form
            cyc = tuple([tuple(k) for k in cyc])
            this_applied = P(inds)
            if tuple(this_applied) not in applied_perms:
                H_N.append(cyc)
        not_equals  = [tuple(Permutation(filled_perm(pi,len(inds)))(inds)) for pi in H_N]
        if len(not_equals) != 0:
            loopperms = not_equals.copy()
        elif len(not_equals) == 0:
            loopperms = all_perms.copy()
        if lreduce:
            tmp = []
            nodes,remainder = tree(inds)
            for loopperm in loopperms:
                grouped = group_vec_by_node(loopperm,nodes,remainder)
                if remainder != None:
                    srted = sorted(grouped[:-1])
                    srted.append(grouped[-1])
                    srted = tuple(srted)
                elif remainder == None:
                    srted = sorted(grouped)
                    srted = tuple(srted)
                if tuple(grouped) == srted:
                    tmp.append(loopperm)
            loopperms=tmp.copy()

        for partition in partitions:
            for fill in loopperms:
                flag,subgroup_fillings = self.check_subgroup_fill(partition,fill,sigma_c_symmetric=sigma_c_symmetric,semistandard=semistandard)
                for subgroup_fill in subgroup_fillings:
                    if subgroup_fill not in subgroup_fills:
                        subgroup_fills.append(subgroup_fill)
                    fills_perpart[tuple(partition)].append(subgroup_fill)
                    try:
                        part_perfill[subgroup_fill].append(tuple(partition))
                    except KeyError:
                        part_perfill[subgroup_fill] = [tuple(partition)]
        all_fills = list(set(subgroup_fills))
        self.fills = sorted(all_fills)
        for sgf, pts in part_perfill.items():
            pts = list(set(pts))
            pts.sort(key=lambda x: x.count(2),reverse=True)
            pts.sort(key=lambda x: tuple([i%2==0 for i in x]),reverse=True)
            pts.sort(key=lambda x: max(x),reverse=True)
            part_perfill[sgf] = pts
        self.fills_per_partition = fills_perpart
        self.partitions_per_fill = part_perfill
