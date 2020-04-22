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
#
# <!-----------------END-HEADER------------------------------------->

import itertools
import numpy as np
import configparser

from distutils.util import strtobool
# Convert a string representation of truth to true (1) or false (0).
# True values are y, yes, t, true, on and 1; false values are n, no, f, false, off and 0.
# Raises ValueError if val is anything else.

# Bispectrum options that don't depend on atom type
untyped_keys = {
    "numtypes":      int,
    "twojmax":       int,
    "rcutfac":       float,
    "rfac0":         float,
    "rmin0":         float,
}

fitkeys = {
    "bzeroflag": strtobool,
    "quadraticflag": strtobool,
    "UseEnergies" : strtobool,
    "UseForces" : strtobool,
    "UseStresses" : strtobool,
}

# Bispectrum options that exist for each atom type
typed_keys = {
    "radelem":  float,
    "type":     str,
    "wj":       float,}

ref_keys={
    "units": str,
    "atom_type": str,}

def _generateblist(twojmax,quadraticflag,alloyflag,numtypes):
    blist = []
    i = 0
    for j1 in range(twojmax+1):
        for j2 in range(j1+1):
            for j in range(abs(j1-j2), min(twojmax,j1+j2)+1,2):
                if j >= j1:
                    i += 1
                    blist.append([i,j1,j2,j])
    if alloyflag:
        blist *= numtypes**3
        blist = np.array(blist).tolist()
    if quadraticflag:
        # Note, combinations_with_replacement precisely generates the upper-diagonal entries we want
        blist += [[i,a,b] for i,(a,b) in enumerate(itertools.combinations_with_replacement(blist, r=2),start=len(blist))]
    return blist

def read_bispec_options(bispec_config, model_config, ref_config):

    bispec_options = {k:tp(bispec_config[k]) for k, tp in untyped_keys.items()}
    bispec_options.update(
        {
            k:[tp(bispec_config[k + str(i + 1)])for i in range(bispec_options["numtypes"])]
               for k,tp in typed_keys.items()
        })
    bispec_options.update({k:tp(model_config[k]) for k, tp in fitkeys.items()})

    bispec_options["alloyflag"] = model_config.get("alloyflag",fallback=0)
    bispec_options["bnames"] = _generateblist(bispec_options["twojmax"],bispec_options["quadraticflag"],bispec_options["alloyflag"],bispec_options["numtypes"])
    bispec_options["n_coeff"] = len(bispec_options["bnames"])
    bispec_options["type_mapping"] = {k: i + 1 for i, k in enumerate(bispec_options["type"])}

    return bispec_options
