# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Charles Sievers (UC Davis, Sandia National Labs)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
#
# #### Additional authors (alphabetical):
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
# <!-----------------END-HEADER------------------------------------->

import gzip
import numpy as np
import pickle
import collections
import datetime

arrayable_keys = (
    'GroupIndex',
    'Index',
    'NumAtoms',
    'eweight',
    'fweight',
    'vweight',
    'QMLattice',
    'Volume',
    'Rotation',
    'Energy',
    'ref_Energy',
    'Stress',
    'ref_Stress',
    'b_sum',
    'vb_sum',
)

##### Functions for packing configs

def pack(configs):

    key_set = set(z for i, x in configs for z in x.keys())

    # Soft error checking: Check that all configs have the same keys and the same type for each key
    type_vals = {}
    error_counts = collections.Counter()
    for i, cf in configs:
        for k in key_set:
            assert (k in cf)
            if k not in type_vals:
                type_vals[k] = type(cf[k])
            else:
                try:
                    assert type(cf[k]) == type_vals[k],\
                        f"Unexpected type: Key: {k}, Expected {type_vals[k]}, got {type(cf[k])}"
                except AssertionError as ae:
                    error_counts.update((k,type_vals[k],type(cf[k])))
                    print("Warning for configuration:",cf["Group"],'/',cf["File"])
                    print(ae)
    if len(error_counts):
        print("Bad type counts:")
        print(dict(error_counts).items(),sep='\n')

    # Put the keys into lists
    key_arrays = {
                  k: [c[k] for i,c in configs]
                    for k in key_set
                  }

    # Pack keys which have uniform representation into numpy arrays
    for key,arr in key_arrays.items():
        if key in arrayable_keys:
            key_arrays[key] = np.asarray(arr)

    return key_arrays

##### Functions for serializing potential

def to_param_string(rcutfac,twojmax,rfac0,rmin0,bzeroflag,quadraticflag,wselfallflag,alloyflag,**unused_options):
    if alloyflag != 0:
        alloyflag_int = 1
    else:
        alloyflag_int = 0
    return f"""
    # required
    rcutfac {rcutfac}
    twojmax {twojmax}

    #  optional
    rfac0 {rfac0}
    rmin0 {rmin0}
    bzeroflag {bzeroflag}
    quadraticflag {quadraticflag}
    #wselfallflag {wselfallflag}
    #alloyflag {alloyflag_int}
    """

def to_coeff_string(coeffs,bispec_options):
    """
    Convert a set of coefficients along with bispec options into a .snapcoeff file
    """
    coeff_names = [[0]]+bispec_options["bnames"] # Includes the offset term, which was not in bnames
    type_names = bispec_options["type"]
    out = f"# fitsnap fit generated on {datetime.datetime.now()}\n\n"
    out += "{} {}\n".format(len(type_names),len(coeff_names))
    for elname, rjval, wjval,column in zip(type_names,bispec_options["radelem"],bispec_options["wj"],coeffs):
        out += "{} {} {}\n".format(elname,rjval,wjval)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval,bname in zip(column,coeff_names))
        out += "\n"
    out+= "\n# End of potential"
    return out
