from fitsnap3lib.lib.sym_ACE.young import * 

# class for defining a coupling tree. It
#  contains tools to sort the tree within 
#  specified orbits of a permutation in S_N.
#  It is often referred to as a representation because,
#  the partitions of S_N are isomorphic to irreps in S_N.
#  the Tree_ID class adopts one of these partitions
class Tree_ID:
    def __init__(self,l,L,maplflag=False):
        self.l=l
        self.maplflag = maplflag
        if self.maplflag:
            self.map_l()
        self.L = L
        self.rank = len(l)
        self.set_sym_block()
        self.set_sigma_c_parts()
        self.orbit = None
        self.orbit_id = None

    def map_l(self):
        unique_ls = [u for u in sorted(list(set(self.l)))]
        lmap = {unique_ls[ind]:ind for ind in range(len(unique_ls))}
        revmapl = {ind:unique_ls[ind] for ind in range(len(unique_ls))}
        thisl = [lmap[i] for i in self.l]
        self.lmap = lmap
        self.revmapl = revmapl
        self.l = thisl

    def set_sym_block(self):
        # sets the highest symmetry partition for a given rank 
        #  it goes in powers of 2^x with some remainder. 
        sym_blocks=  {  4:((4,),),
                        5:((4,1),),
                        6:((4,2),),
                        7:((4,2,1),),
                        8:((8,),),
                      }
        self.sym_block = sym_blocks[self.rank][0]

    def return_subtree_l_ID(self,l,L=None,L_R=0):
        if L == None:
            L = tree_l_inters(l,L_R)
        if self.rank == 4:
            if L[0] == L[1]:
                L_id = tuple([int(k) for k in range(len(set(L)))])
            elif L[0] != L[1]:
                L_id_tmp = tree_l_inters(l,L_R)[0]
                L_id = tuple([int(k) for k in range(len(set(L_id_tmp)))])
            subtree_lL = (tuple(l),L_id,L_R)
        elif self.rank == 5:
            #NOTE this will need to be adjusted for L_R !=0
            L_id_tmp = None
            if L[0] == L[1]:
                for linter in  tree_l_inters(l,L_R):
                    if L[0] == L[1]:
                        L_id_tmp = linter
                        break
            elif L[0] != L[1]:
                for linter in  tree_l_inters(l,L_R):
                    if L[0] != L[1]:
                        L_id_tmp = linter
                        break
                 
            L_id = tuple([int(k) for k in range(len(set(L_id_tmp[:-1])))])
            if l[4] == L_id_tmp[2]:
                L_id_2 = (0,l[4])
            elif l[4] != L_id_tmp[2]:
                L_id_2 = (1,l[4])
            subtree_1 = (tuple(sorted(l[:-1])), L_id)
            subtree_2 = L_id_2
            subtree_lL = (subtree_1,subtree_2)
        self.subtree_lL = subtree_lL
        #return subtree_lL
        


    def set_sigma_c_parts(self):
        ysgi = Young_Subgroup(self.rank)
        sigma_c_parts = ysgi.sigma_c_partitions(max_orbit=self.rank)
        sigma_c_parts.sort(key=lambda x: x.count(2),reverse=True)
        self.sigma_c = sigma_c_parts[0]
        sigma_c_parts.sort(key=lambda x: tuple([i%2==0 for i in x]),reverse=True)
        sigma_c_parts.sort(key=lambda x: max(x),reverse=True)
        self.sigma_c_parts = sigma_c_parts
        self.ysg = ysgi
        # generate leaf nodes structure for the binary tree
        nodes,remainder = tree(self.l)
        self.nodes = nodes
        self.remainder = remainder

    def return_leaf_l_ID(self):
        #if len(self.l) < 6:
        leaf_l = group_vec_by_node(self.l , self.nodes , self.remainder)
        leaf_id = [(tuple(sorted(lli)),Li) for lli,Li in zip(leaf_l,self.L)]
        leaf_only = [tuple(sorted(lli)) for lli in leaf_l]
        leaf_id = tuple(sorted(leaf_id))
        self.ltree = leaf_id
        self.leaf_only_l = leaf_only
        self.l_leaf_sym = [len(set(ln)) == 1 for ln in leaf_l]
        return leaf_id
    
    def return_orbit_l_ID(self,orbit=None):
        if orbit == None and self.orbit == None:
            orbit = self.sym_block
            self.orbit = orbit
        elif orbit != None:
            self.orbit = orbit
        else:
            orbit = self.orbit
        orb_l = group_vec_by_orbits(self.l, orbit)
        orb_id = []
        orb_only = []
        Lcount = 0
        for lorbit in orb_l:
            L_count_add = math.ceil(len(lorbit)/2)
            orbit_L = self.L[Lcount:Lcount + L_count_add] 
            Lcount += L_count_add
            orbit_nodes, orbit_remainder = tree(lorbit)
            leaf_l_perorb_l = group_vec_by_node(lorbit , orbit_nodes , orbit_remainder)
            #orbit_leaf_id = tuple(sorted(leaf_l_perorb))
            #orbit_leaf_id = sorted(orbit_leaf_id)
            orbit_leaf_add = tuple(sorted(lorbit))
            #leaf_l_perorb_l = group_vec_by_node(lorbit , orbit_nodes , orbit_remainder)
            leaf_l_perorb = group_vec_by_node(lorbit , orbit_nodes , orbit_remainder)
            #leaf_l_perorb = (tuple(sorted(leaf_l_perorb_l)),tuple(sorted(orbit_L)))
            #print ('leafl per orb',leaf_l_perorb)
            orbit_leaf_id = leaf_l_perorb
            #orbit_leaf_id = tuple(sorted(leaf_l_perorb))
            #orbit_leaf_id = sorted(orbit_leaf_id)
            orbit_leaf_add =tuple(sorted(lorbit))
            if self.maplflag:
                orbit_leaf_add = tuple([self.revmapl[ikl] for ikl in orbit_leaf_add])
            orb_id.append(tuple(orbit_leaf_add))
        orb_id = tuple(orb_id)
        self.orbit_id = orb_id
        self.l_orb_sym = [len(set(ln)) == 1 for ln in orb_l]
        return orb_id
    
    def return_orbit_nl_ID(self,n,orbit =None):
        #unique_ns = [u for u in sorted(list(set(nin)))]
        #nmap = {unique_ns[ind]:ind for ind in range(len(unique_ns))}
        #revmap = {ind:unique_ns[ind] for ind in range(len(unique_ns))}
        #n = [nmap[i] for i in nin]
        if self.orbit == None:
            new_lid = self.return_orbit_l_ID(orbit=orbit)
            #new_lid = tuple([o[0] for o in new_lid])
        if orbit != None:
            self.orbit = orbit
            new_lid = self.return_orbit_l_ID(orbit=orbit)
            #new_lid = tuple([o[0] for o in new_lid])
        #new_lid = [o[0] for o in self.orbit_id]
        orb_n = group_vec_by_orbits(n, self.orbit)
        onl_id = []
        degen_id = []
        for ol,on in zip(new_lid,orb_n):
            #ol,oL = olL
            itups = [(nii,lii) for nii,lii in zip(on,ol)]
            itups = sorted(itups)
            degen_id.append(tuple([(nii,lii) for nii,lii in zip(sorted(on),sorted(ol))]))
            orb_inner = tuple([ii[0] for ii in itups])
            id_inner = tuple([ii[1] for ii in itups])
            #onl=tuple([orb_inner,id_inner,orb_inner_L])
            onl=tuple([orb_inner,id_inner])
            onl_id.append(onl)
        onl_id = sorted(onl_id)
        degen_id = tuple(sorted(degen_id))
        self.degen_id = degen_id
        onl_id.sort(key=lambda x: len(x),reverse = True)
        onl_id.sort(key=lambda x: len(x[0]),reverse = True)
        return tuple(onl_id)

    def return_orbit_munl_ID(self,muin,nin,orbit =None):
        unique_ns = [u for u in sorted(list(set(nin)))]
        nmap = {unique_ns[ind]:ind for ind in range(len(unique_ns))}
        revmap = {ind:unique_ns[ind] for ind in range(len(unique_ns))}
        n = [nmap[i] for i in nin]
        unique_mus = [u for u in sorted(list(set(muin)))]
        mumap = {unique_mus[ind]:ind for ind in range(len(unique_mus))}
        revmapmu = {ind:unique_mus[ind] for ind in range(len(unique_mus))}
        mu = [mumap[i] for i in muin]
        if self.orbit == None:
            new_lid = self.return_orbit_l_ID(orbit=orbit)
        if orbit != None:
            self.orbit = orbit
            new_lid = self.return_orbit_l_ID(orbit=orbit)
        new_lid = self.orbit_id
        orb_n = group_vec_by_orbits(n, self.orbit)
        orb_mu = group_vec_by_orbits(mu, self.orbit)
        onl_id = []
        degen_id = []
        for ol,on,omu in zip(new_lid,orb_n,orb_mu):
            itups = [(muii,nii,lii) for muii,nii,lii in zip(omu,on,ol)]
            itups = sorted(itups)
            degen_id.append(tuple([(muii,nii,lii) for muii,nii,lii in zip(sorted(omu),sorted(on),sorted(ol))]))
            #unmapped indices for speed increase
            orb_inner = tuple([ii[1] for ii in itups])
            orb_outer = tuple([ii[0] for ii in itups])
            #mapped indices for validation
            #orb_inner = tuple([revmap[ii[1]] for ii in itups])
            #orb_outer = tuple([revmapmu[ii[0]] for ii in itups])
            id_inner = tuple([ii[2] for ii in itups])
            onl=tuple([orb_outer,orb_inner,id_inner])
            onl_id.append(onl)
        onl_id = sorted(onl_id)
        degen_id = tuple(sorted(degen_id))
        self.degen_id = degen_id
        onl_id.sort(key=lambda x: len(x),reverse = True)
        onl_id.sort(key=lambda x: len(x[0]),reverse = True)
        return tuple(onl_id)

def check_recursion_related(l,L,L_R=0,recursion_related=[]):
    rank=len(l)
    all_inters = tree_l_inters(l)
    nodes,remainder = tree(l)
    lnodes = group_vec_by_node(l,nodes,remainder)
    inters = parity_filter(lnodes,l)
    if rank == 4:
        L1 = L[0]
        L2 = L[1]
        if L_R == 0:
            assert L1==L2, "if L1 != L2 for L_R = 0 , then the wigner symbol is invalid"
            L1L2_min = inters[0]
            conditions =  L1 - L1L2_min[0] == L2 - L1L2_min[1]
            if conditions or (l,L) in recursion_related:
                recursion_related.extend( [(l,inter) for inter in inters])
                
    elif rank == 5:
        L1=L[0]
        L2=L[1]
        L3=L[2]
        if L_R == 0:
            assert L3==l[5], "if L3 != l5 for L_R = 0 , then the rank 5 wigner symbol is invalid"
            L1L2L3_min = inters[0]
            condition0 = L1 == L1L2L3_min[0] and L2 == L1L2L3_min[1] #or ... #NOTE what about equivalent perms?
            condition1 = L1 - L1L2L3_min[0] == L2 - L1L2L3_min[1]
            condition2 = L1 - L1L2L3_min[0] != L2 - L1L2L3_min[1]
            if condition0 or condition1 or condition2:
                recursion_related.extend( [(l,inter) for inter in inters])
                return True,recursion_related



def grow_recursion_related(l,L_R=0):
    if type(l) == list:
        l = tuple(l)
    nrecurr = 20
    rank=len(l)
    all_inters = tree_l_inters(l)
    nodes,remainder = tree(l)
    lnodes = group_vec_by_node(l,nodes,remainder)
    inters = parity_filter(lnodes,all_inters)
    if rank == 4:
        if L_R == 0:
            recursion_related = {'co':[]}
            for Li in inters:
                L1 = Li[0]
                L2 = Li[1]
                L1L2_min = inters[0]
                conditions =  L1 - L1L2_min[0] == L2 - L1L2_min[1]
                if conditions:
                    recursion_related['co'].append((l,Li))
                
    elif rank == 5:
        recursion_related = {'co':[],'c12':[],'c1':[]}
        for Li in inters:
            L1=Li[0]
            L2=Li[1]
            L3=Li[2]
            if L_R == 0:
                L1L2L3_min = inters[0]
                condition0 = L1 == L1L2L3_min[0] and L2 == L1L2L3_min[1] or L1 == L1L2L3_min[1] and  L2 == L1L2L3_min[0]
                condition1a = L1 - L1L2L3_min[0] == L2 - L1L2L3_min[1]
                condition1b = L1 - L1L2L3_min[0] != 0 or L2 - L1L2L3_min[1] !=0
                condition1c = L1 == L2
                condition1 = condition1a and condition1b or condition1c
                condition2 = L1 - L1L2L3_min[0] != L2 - L1L2L3_min[1] and L1 != L1L2L3_min[0]# and L2 != L1L2L3_min[1]

                if condition0:
                    for nk1 in range(nrecurr):
                        Li1 = (Li[0] + nk1,Li[1],Li[2])
                        Li2 = (Li[0] ,Li[1]+nk1,Li[2])
                        Li1m = (Li[0] - nk1,Li[1],Li[2])
                        Li2m = (Li[0] ,Li[1]-nk1,Li[2])
                        if Li1 in inters:
                            if (l,Li1) not in recursion_related['co']:
                                recursion_related['co'].append((l,Li1))
                        if Li2 in inters:
                            if (l,Li2) not in recursion_related['co']:
                                recursion_related['co'].append((l,Li2))
                        if Li1m in inters:
                            if (l,Li1) not in recursion_related['co']:
                                recursion_related['co'].append((l,Li1m))
                        if Li2m in inters:
                            if (l,Li2) not in recursion_related['co']:
                                recursion_related['co'].append((l,Li2m))
                if condition1 and not condition0:
                    #for nk1,nk2 in itertools.product(range(nrecurr),range(nrecurr)):
                    for nk1 in range(nrecurr):
                        #Li1 = (Li[0] + nk1,Li[1] +nk2,Li[2])
                        Li1 = (Li[0] +nk1 ,Li[1]+nk1,Li[2])
                        if Li1 in inters:
                            if (l,Li1) not in recursion_related['co'] + recursion_related['c12']:
                                recursion_related['c12'].append((l,Li))
                            #if (l,Li2) not in recursion_related['co'] + recursion_related['c12']:
                            #   recursion_related['c12'].append((l,Li2))
                            
                if condition2 and not condition1 and not condition0:
                    for nk1,nk2 in itertools.product(range(nrecurr),range(nrecurr)):
                        Li1 = (Li[0] +nk1 ,Li[1]+nk2,Li[2])
                        if Li1 in inters:
                            if (l,Li1) not in recursion_related['co'] + recursion_related['c12'] + recursion_related['c1']:
                                recursion_related['c1'].append((l,Li1))
    return recursion_related


#related = grow_recursion_related([1,1,1,1,2])
#print (related)
#related = grow_recursion_related([1,1,2,2,4])
#print (related)
