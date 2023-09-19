from fitsnap3lib.lib.sym_ACE.young import * 

def check_equal_nodes(nd1,nd2):
    equal = False
    if nd1.val_tup == None:
        nd1.current_val_tup()
    if nd2.val_tup == None:
        nd2.current_val_tup()
    equal =  nd1.val_tup == nd2.val_tup

class Node:
    def __init__(self,i):
        self.val = i
        self.desc_rank = None
        self.val_tup = None
        self.children = [None,None]
        self.leaves = []
        self.lft = self.children[0]
        self.rght = self.children[1]
        self.is_sorted = False
        self.is_symmetric = False
        self.has_children = False
        self.parent = None

    def set_parent(self,parent):
        self.parent = parent

    def set_leaves(self,leaves):
        self.leaves = leaves

    def update_leaves(self,leaf):
        self.leaves.append(leaf)

    def check_sorted(self,sort_depth=1):
        self.is_sorted = False
        if sort_depth == 1:
            if self.lft == None and self.rght == None:
                self.is_sorted = True
            elif self.lft == None and self.rght != None:
                self.is_sorted = False
            elif self.lft != None and self.rght == None:
                self.is_sorted = False
            elif self.lft != None and self.rght != None:
                if self.lft.val <= self.rght.val:
                    self.is_sorted = True
                elif self.lft.val > self.rght.val:
                    self.is_sorted = False
            returnval = self.is_sorted
        elif sort_depth == 2:
            assert self.lft != None and self.rght != None,"must have parent nodes to compare children"
            self.lft.current_val_tup()
            lv = self.lft.val_tup[1]
            self.rght.current_val_tup()
            rv = self.rght.val_tup[1]
            if self.lft.val_tup < self.rght.val_tup:
                if lv[0] <= lv[1] and rv[0] <= rv[1]:
                    self.is_sorted = True
                else:
                    if lv[0] > lv[1]:
                        self.lft.flip_children()
                        self.lft.current_val_tup()
                        lv = self.lft.val_tup[1]
                        self.lft.is_sorted = True
                    if rv[0] > rv[1]:
                        self.rght.flip_children()
                        self.rght.current_val_tup()
                        rv = self.rght.val_tup[1]
                        self.rght.is_sorted = True
                    self.is_sorted = True
            elif lv > rv:
                self.is_sorted = False
                if lv[0] > lv[1]:
                    self.lft.flip_children()
                    self.lft.current_val_tup()
                    lv = self.lft.val_tup[1]
                    self.lft.is_sorted = self.lft.check_sorted()
                if rv[0] > rv[1]:
                    self.rght.flip_children()
                    self.rght.current_val_tup()
                    rv = self.rght.val_tup[1]
                    self.rght.is_sorted = self.rght.check_sorted()
            returnval = self.lft.is_sorted and self.rght.is_sorted and self.is_sorted
        else:
            raise ValueError('cant go that deep yet')
        return returnval

    def return_children_vals(self,depth=1):
        if depth==1:
            vals = (self.lft.val,self.rght.val)
        elif depth==2:
            self.lft.current_val_tup()
            lv = self.lft.val_tup[1]
            self.rght.current_val_tup()
            rv = self.rght.val_tup[1]
            vals = lv + rv
        return vals
            

    def set_children(self,children,set_sorted=False,sort_depth=1):
        if type(children) == list:
            children = tuple(children)
        assert len(children) == 2, "list of children must be of length 2 for binary tree"
        if children[0] != None and children[1] != None:
            children[0].set_parent(self)
            children[1].set_parent(self)
            self.children[0] = children[0]
            self.children[1] = children[1]
            child_val_list = [children[0].val,children[1].val]
            self.check_sorted(sort_depth)
            self.has_children = True
            if set_sorted:
                if self.is_sorted:
                    if sort_depth == 1:
                        self.lft = self.children[0]
                        self.rght = self.children[1]
                    elif sort_depth == 2:
                        self.lft = self.children[0]
                        self.rght = self.children[1]
                        if self.lft.is_sorted:
                            self.lft.lft = self.lft.children[0]
                            self.lft.rght = self.lft.children[1]
                        elif not self.lft.is_sorted:
                            self.lft.lft = self.lft.children[1]
                            self.lft.rght = self.lft.children[0]
                            self.lft.is_sorted = True
                        if self.rght.is_sorted:
                            self.rght.lft = self.rght.children[0]
                            self.rght.rght = self.rght.children[1]
                        elif not self.rght.is_sorted:
                            self.rght.lft = self.rght.children[1]
                            self.rght.rght = self.rght.children[0]
                            self.rght.is_sorted = True

                elif not self.is_sorted:
                    if sort_depth == 1:
                        self.lft = self.children[1]
                        self.rght = self.children[0]
                    elif sort_depth == 2:
                        self.lft = self.children[1]
                        self.rght = self.children[0]
                        self.lft.is_sorted = self.lft.check_sorted()
                        self.rght.is_sorted =self.rght.check_sorted()
                        if self.lft.is_sorted:
                            self.lft.lft = self.lft.children[0]
                            self.lft.rght = self.lft.children[1]
                        elif not self.lft.is_sorted:
                            self.lft.lft = self.lft.children[1]
                            self.lft.rght = self.lft.children[0]
                            self.lft.is_sorted = True
                        if self.rght.is_sorted:
                            self.rght.lft = self.rght.children[0]
                            self.rght.rght = self.rght.children[1]
                        elif not self.rght.is_sorted:
                            self.rght.lft = self.rght.children[1]
                            self.rght.rght = self.rght.children[0]
                            self.rght.is_sorted = True
                    self.is_sorted = True

    def update_children(self,set_sorted=False,sort_depth=1):
        if self.lft !=None and self.rght != None:
            children = [self.lft,self.rght]
            self.set_children(children,set_sorted,sort_depth=sort_depth)

    def current_val_tup(self):
        this_tup =(self.val,((self.children[0].val), (self.children[1].val)))
        self.val_tup = this_tup
        return this_tup

    def full_tup(self):
        assert self.desc_rank != None, "you first need to add branches/other nodes to print the tree"

        ysgi = Young_Subgroup(self.desc_rank)
        sigma_c_parts = ysgi.sigma_c_partitions(max_orbit=self.desc_rank)
        sigma_c_parts.sort(key=lambda x: x.count(2),reverse=True)
        sigma_c_parts.sort(key=lambda x: tuple([i%2==0 for i in x]),reverse=True)
        sigma_c_parts.sort(key=lambda x: max(x),reverse=True)
        valid_coupling_partitions = sigma_c_parts[:-1]

        if self.desc_rank == 4:
            tree_id = ( (self.val,),  (self.lft.val , (self.lft.lft.val, self.lft.rght.val)  ) ,   (self.rght.val, (self.rght.lft.val, self.rght.rght.val) ) )
            leafs = [ self.lft.lft.val, self.lft.rght.val, self.rght.lft.val, self.rght.rght.val   ]
            sub_1_sym = self.lft.lft.val == self.lft.rght.val 
            sub_2_sym = self.rght.lft.val == self.rght.rght.val
            orbit_l_sym = (self.lft.lft.val, self.lft.rght.val) == ( self.rght.lft.val, self.rght.rght.val) 
            root_L_sym = self.lft.val == self.rght.val
            sym_map = {True:'s',False:'a'}
            root_sym = root_L_sym and orbit_l_sym
            sym_tup = tuple([sym_map[root_sym],sym_map[sub_1_sym],sym_map[sub_2_sym]])

            sym_str = \
"""
     %s
     %s       %s
""" % sym_tup
            self.sym_tup = sym_tup
            self.sym_str = sym_str

            print_str =  \
"""
     %s
     %s       %s
   %s   %s   %s   %s
"""
            L_lv1 = [self.lft.val, self.rght.val]
            rootval = self.val 
            print_tup = tuple( [rootval] + L_lv1 + leafs )
            part_by_part = {}
            sorted_partitions = [] 
            for part in valid_coupling_partitions:
                leaf_orbs = group_vec_by_orbits(leafs, part)
                sorted_in_partition = is_row_sort(leaf_orbs)
                if sorted_in_partition:
                    sorted_partitions.append(part)
                elif not sorted_in_partition:
                    continue
            
        elif self.desc_rank == 5:
            tree_id = ( (self.val,),  (self.lft.val,  (self.lft.lft.val, (self.lft.lft.lft.val, self.lft.lft.rght.val)  ) , (self.lft.rght.val, (self.lft.rght.lft.val , self.lft.rght.rght.val) ) ) , (self.rght.val,))
            sub_1_sym = self.lft.lft.lft.val == self.lft.lft.rght.val
            sub_2_sym = self.lft.rght.lft.val == self.lft.rght.rght.val
            sub_3_sym = self.lft.lft.val == self.lft.rght.val and (self.lft.lft.lft.val, self.lft.lft.rght.val) == (self.lft.rght.lft.val, self.lft.rght.rght.val)
            leafs = [self.lft.lft.lft.val, self.lft.lft.rght.val, self.lft.rght.lft.val , self.lft.rght.rght.val, self.rght.val]
             
            root_sym = sub_1_sym and sub_2_sym and sub_3_sym and self.lft.val == self.rght.val and len(set(leafs)) == 1
            sym_map = {True:'s',False:'a'}
            sym_tup = tuple([sym_map[root_sym], sym_map[sub_3_sym], sym_map[sub_1_sym],sym_map[sub_2_sym]])
            sym_str = \
"""
     %s
     %s      
  %s    %s
""" % sym_tup
            self.sym_tup = sym_tup
            self.sym_str = sym_str
            print_str = \
"""
        %s
     %s          %s
     %s       %s
   %s   %s   %s   %s
"""
            # with removed l_i on the different level for printing purposes
            leafs_lv0 = [self.lft.lft.lft.val, self.lft.lft.rght.val, self.lft.rght.lft.val , self.lft.rght.rght.val]
            L_lv2 = [self.lft.lft.val, self.lft.rght.val]
            L_lv1 = [self.lft.val]
            # with l_i on the same level for printing purposes
            lL_lv1 = [self.lft.val,self.rght.val]
            rootval = self.val
            print_tup = tuple( [rootval] + lL_lv1 + L_lv2 + leafs_lv0 )
            part_by_part = {}
            sorted_partitions = []
            for part in valid_coupling_partitions:
                leaf_orbs = group_vec_by_orbits(leafs, part)
                sorted_in_partition = is_row_sort(leaf_orbs)
                if sorted_in_partition:
                    sorted_partitions.append(part)
                elif not sorted_in_partition:
                    continue

        elif  self.desc_rank == 6:
            tree_id = ( (self.val,),  (self.lft.val, ( self.lft.lft.val, ( self.lft.lft.lft.val, self.lft.lft.rght.val)  ) , (self.lft.rght.val, (self.lft.rght.lft.val, self.lft.rght.rght.val) ) ), (self.rght.val,  (self.rght.lft.val, self.rght.rght.val ) ) )

            leafs = [ self.lft.lft.lft.val, self.lft.lft.rght.val, self.lft.rght.lft.val, self.lft.rght.rght.val, self.rght.lft.val,self.rght.rght.val ]
            internals = [self.lft.lft.val, self.lft.rght.val, self.rght.val, self.lft.val]

            leafs_lv0 = [self.lft.lft.lft.val, self.lft.lft.rght.val, self.lft.rght.lft.val, self.lft.rght.rght.val]
            lv1 = [self.lft.val,self.rght.val]
            lv2 = [self.lft.lft.val, self.lft.rght.val, self.rght.lft.val, self.rght.rght.val]
            rootval = self.val
            self.sym_tup = None
            print_tup = tuple( [rootval] + lv1 + lv2 + leafs_lv0 )
            print_str_template = \
"""
        0
     1          2
     2       3      4       5
   6   7   8   9
"""
            print_str = \
"""
        %s
     %s          %s
     %s       %s      %s       %s
   %s   %s   %s   %s
"""

        self.printable = print_str % print_tup
        self.print_str = print_str
        self.sym_str = None
        if self.is_sorted:
            self.tree_id = tree_id
        else:
            self.tree_id = tree_id
        
    def flip_children(self):
        self.children.reverse()
        self.set_children(self.children)
        
    def sort_tree(self):
        if self.is_sorted:
            return
        else:
            self.children.reverse()
            self.set_children(self.children)


def build_tree(l,L,L_R):
    assert type(l) == list or type(l) == tuple, "convert l to list or tuple"
    assert type(L) == list or type(L) == tuple, "convert L to list or tuple"
    rank = len(l)

    def rank_2_binary(l_parent,l_left,l_right):
        if type(l_parent) == int:
            root2 = Node(l_parent)
        else:
            root2 = l_parent
        root2.lft = Node(l_left)
        root2.rght = Node(l_right)
        root2.update_children(set_sorted=True)
        leaves = (root2.lft.val,root2.rght.val)
        return root2,leaves


    if rank == 4:
        root = Node(L_R)
        this_tree,left_leaves =rank_2_binary(L[0],l[0],l[1])
        root.lft,left_leaves = rank_2_binary(L[0],l[0],l[1])
        root.rght,right_leaves = rank_2_binary(L[1],l[2],l[3])
        root.update_children(set_sorted=True,sort_depth=2)
        test_leaves = root.return_children_vals(depth=2)
        root.set_leaves(list(test_leaves))
    elif rank == 5: 
        root = Node(L_R)
        root.lft = Node(L[2])
        root.rght = Node(l[4])
        root.lft.lft,left_leaves = rank_2_binary(L[0],l[0],l[1])
        root.lft.rght,right_leaves = rank_2_binary(L[1],l[2],l[3])
        root.lft.update_children(set_sorted=True,sort_depth=2)
        test_leaves = root.lft.return_children_vals(depth=2)
        #root.set_leaves(list(test_leaves)+(l[4],))
        root.set_leaves(list(test_leaves)+[l[4]])

    elif rank == 6:
        root = Node (L_R)
        root.lft = Node(L[3])
        root.rght,leaves_3 = rank_2_binary(L[2],l[4],l[5])
        root.rght.update_children(set_sorted=True)
        root.update_children(set_sorted=True)
        root.lft.lft, leaves_1 = rank_2_binary(L[0],l[0],l[1])
        root.lft.rght, leaves_2 = rank_2_binary(L[1],l[2],l[3])
        root.lft.update_children(set_sorted=True,sort_depth=2)
        test_leaves = root.lft.return_children_vals(depth=2)
        root.set_leaves(list(test_leaves + leaves_3))
    root.desc_rank = rank
    return root

#ntst = [2,1,1,2]
#ltst = [1,2,1,2]
#l = [(li,ni) for ni,li in zip(ntst,ltst)]

#L_R=0
#L=[0,0]
#tree_i =  build_tree(l,L,L_R)
#ttup = tree_i.full_tup()
#print ('id',tree_i.tree_id)
#print ('full_tup',ttup)
#print ('leaves in:',l)
#print ('leaves out:',tree_i.leaves)


#print (tree_i.printable)
