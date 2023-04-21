# global settings for the sym_ACE library
from fitsnap3lib.lib.sym_ACE.sym_ACE_settings import *
import sys

try:
    flag = sys.argv[1]
except IndexError:
    flag = None
if flag != None:
    #   list of ranks
    print ('!!! IMPORTANT !!! lmax per rank:')
    for rank in ranks:
        print(rank,lmax_dict_G[rank])
    print ('Attempting to use descriptors with larger lmax values than this per rank will result in KeyError')
    print ('  To use larger values, edit lmax_dict_G in the sym_ACE_settings.py file. Using larger values \n may result in long times to load libraries.')

    print ('Prompting generation for first library of coupling coefficients:')
    from wigner_couple import *
    print (global_ccs)
    print ('sent')
