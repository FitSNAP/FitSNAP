from lammps import lammps
import numpy as np

lmp = lammps()
lmp.commands_string("""
newton off
region box block -2 2 -2 2 -2 2
lattice fcc 1.0
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style zero 1.0 full ncoeff
pair_coeff * *
run 0 post no""")

# look up the neighbor list
nlidx = lmp.find_pair_neighlist('zero')
nl = lmp.numpy.get_neighlist(nlidx)
tags = lmp.extract_atom('id')
print("full neighbor list with {} entries".format(nl.size))
# print neighbor list contents
for i in range(0,nl.size):
    idx, nlist  = nl.get(i)
    #print("\natom {} with ID {} has {} neighbors:".format(idx,tags[idx],nlist.size))
    if nlist.size > 0:
        for n in np.nditer(nlist):
            pass
            #print("  atom {} with ID {}".format(n,tags[n]))