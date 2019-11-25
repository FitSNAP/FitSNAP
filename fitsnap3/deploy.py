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

import os
import copy

import multiprocessing

import tqdm

try:
    import lammps
except ImportError:
    raise ImportError("Could not find LAMMPS.py. Install LAMMPS and link.")

from . import runlammps

DEFAULT_CHUNKSIZE = 10
DEFAULT_LAMMPS_LOG = False

def check_lammps(lammps_noexceptions):
    lmp = lammps.lammps()
    if not (lmp.has_exceptions or lammps_noexceptions):
        raise Exception("Fitting interrupted! LAMMPS not compiled with C++ exceptions handling enabled")
    del lmp

def compute_bispec_datasets(input_configs, bispec_options, n_procs, chunk_size=DEFAULT_CHUNKSIZE,  log=DEFAULT_LAMMPS_LOG):
    if n_procs == 1:
        results = compute_single(input_configs, bispec_options, log=log)
    else:
        results = compute_multi(input_configs, bispec_options, n_procs, chunk_size, log=log)
    assert all(i==j==data["Index"] for i,(j,data) in enumerate(results)),\
        "Configurations dropped or out of order"

    return results

### Compute using single process

def compute_single(data_dict, bispec_options, log=DEFAULT_LAMMPS_LOG):
    if log:
        lmp = lammps.lammps(cmdargs=["-screen","none"])
    else:
        lmp = lammps.lammps(cmdargs=["-screen","none","-log","none"])
    all_data = []
    for i,data in tqdm.tqdm(enumerate(data_dict),desc="Configs",total=len(data_dict),disable=(not bispec_options["verbosity"]), ascii=True):
        data = copy.deepcopy(data)
        comp = runlammps.compute_lammps(lmp=lmp,
                                 data=data,
                                 bispec_options=bispec_options)
        data.update(comp)
        all_data.append((i,data))
    return all_data

### Compute using multiple process and multiprocessing module

lmplogGLOBAL = False
def initializer():
    global lmpGLOBAL
    #print("Initializing LAMMPS in process", os.getpid())
    logname = "log-pid{}.lammps".format(os.getpid())
    if lmplogGLOBAL:
        lmpGLOBAL = lammps.lammps(cmdargs=["-screen","none","-log",logname])
    else:
        lmpGLOBAL = lammps.lammps(cmdargs=["-screen","none","-log","none"])

chunkGLOBAL = None
bispecGLOBAL = None
def compute_partial(inputs):
    i, data = inputs
    data = copy.deepcopy(data)
    comp = runlammps.compute_lammps(lmp=lmpGLOBAL,
                                    data=data,
                                    bispec_options=bispecGLOBAL
                                    )
    data.update(comp)
    return (i, data)

def compute_multi(data_dict, bispec_options, n_procs, chunk_size=DEFAULT_CHUNKSIZE,log=DEFAULT_LAMMPS_LOG):
    global chunkGLOBAL, bispecGLOBAL, lmplogGLOBAL

    chunkGLOBAL = chunk_size
    bispecGLOBAL = bispec_options
    lmplogGLOBAL = log
    fork = multiprocessing.get_context(method='fork')
    with fork.Pool(n_procs, initializer=initializer) as pool:
            results = pool.imap_unordered(compute_partial, enumerate(data_dict), chunksize=chunk_size)#,total=len(data_dict))
            results = list(tqdm.tqdm(results,desc="Configs",total=len(data_dict),disable=(not bispec_options["verbosity"]), ascii=True))

    results = list(sorted(results,key=lambda pair:pair[0]))

    return results
