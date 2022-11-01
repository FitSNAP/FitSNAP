# global settings for the sym_ACE library
import sys
# Name of the directory the repo is stored in:
import fitsnap3lib
topfile = fitsnap3lib.__file__
top_dir = topfile.split('/__')[0]
lib_path = '%s/lib/sym_ACE/lib' % top_dir

#-------------------------------
# Coupling coefficient settings
#-------------------------------
ranks = range(1,7)
# lmax per rank
lmax_dict_G = {1:0 , 2:8, 3:6, 4:4, 5:3, 6:2}
# lmax for underlying traditional wigner-3j symbols (will impose limits on lmax_dict_G)
lmax_traditional=10
# Flag to generate library of CG coefficients as well
cglib = False
# Flag to generate generalized wigner coefficients for L_R = 1 reduced reps (vector-like descriptors)
gen_LR1 = False
