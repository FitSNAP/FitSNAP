"""
Python script for performing a fit and immediately calculating test errors after the fit.

Test errors are reported for MAE energy (eV/atom) and MAE force (eV/Angstrom), if using LAMMPS 
metal units.

Usage:

    python example.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example-nodump.in
"""

import numpy as np
import os, psutil
import resource
import time
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"main script comm: {comm}")

# Check out how to split this communicator among all processes:
# https://www.codingame.com/playgrounds/349/introduction-to-mpi/splitting
# E.g. to split the communicator in half see below.
# E.g. with np 4, this will create 2 groups with ranks (0,1) and (2,3).
# In the comm_split communicator, this will make rank 1 be rank 0, and rank 2 be rank 0, due to key.
# Can also set different settings in each communicator.
if rank < size//2:
    color = 10
    key = -rank
    twojmax = 2

else:
    color = 20
    key = +rank
    twojmax = 6

print(f"proc {rank} color {color} key {key}")
# comm2 is a split communicator involving colors of this group
comm_split = comm.Split(color=color, key=key)
print(f"comm size {comm.Get_size()} comm_split size: {comm_split.Get_size()}")

# Create an input dictionary containing settings.

data = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": twojmax, #6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "../../Ta_Linear_JCP2014/JSON"
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 1,
    "dump_truth": 1,
    "dump_weights": 1,
    "dump_dataframe": 1
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0     1.00E-08        1.00E-08        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.00E-09        1.00E-09        1.00E-09",
    "GSF_110"       :  "1.0    0.0      100             1               1.00E-08",
    "GSF_112"       :  "1.0    0.0      100             1               1.00E-08",
    "Liquid"        :  "1.0    0.0       4.67E+02        1               1.00E-08",
    "Surface"       :  "1.0    0.0       100             1               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Declare some settings for this fitsnap object.
# These settings determine:
# 1. style of input (file or dict)

# E.g. create instance with settings from filename, using global comm
#filename = "../../Ta_Linear_JCP2014/Ta-example.in"
#snap = FitSnap(filename, comm=comm, arglist=["--overwrite"])

# E.g. create two instances:
"""
snap = FitSnap(filename, comm=comm_split, arglist=["--overwrite"])
snap2 = FitSnap(filename, comm=comm_split, arglist=["--overwrite"])
"""

# E.g. create an single instance with comm split on each group:
#snap = FitSnap(filename, comm=comm_split, arglist=["--overwrite"])

# E.g. create instance with a dictionary input (no file)
#snap = FitSnap(data, comm=comm, arglist=["--overwrite"])

# E.g. create an instance with indict and split communicator:
# This should increase memory usage by number of comm groups, since new shared arrays will be made for each comm.
snap = FitSnap(data, comm=comm_split, arglist=["--overwrite"])

# tell ParallelTool not to create SharedArrays
#pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
#pt.check_fitsnap_exist = False
# tell FitSNAP not to delete the data object after processing configs
#snap.delete_data = False

snap.scrape_configs()
# After this point, there is already a `data` dictionary in the `snap` instance, so no need to reread everything with 
# the second instance. Just inject the data dict into the second instance (also along with parallel tools too maybe?).
#snap2.scrape_configs()
snap.process_configs()

# Stop here to check the memory
#process = psutil.Process()
#print(f"--- ptrank {snap.pt._rank} pid {psutil.Process(os.getpid())} procmem: {process.memory_info().rss}")

# Sleep to check top:
#for i in range(0,100000000000000000):
#    b = i+1

# Let's test shared array functionality.
"""
if (rank == 0):
    snap.pt.shared_arrays['a'].array[0,0] = 1e10
snap.pt._comm.Barrier()
if (rank == 1):
    # There is no need to have all instances read the same data...
    # Could inject data from once instance into another to save memory on a node.
    # Both instances here used all procs.
    #print(snap.pt.shared_arrays['a'].array)
    pass
print(f"--- rank {rank} A: {snap.pt.shared_arrays['a'].array}")
"""
snap.pt.all_barrier()
#snap2.pt.all_barrier()
snap.perform_fit()
##snap2.perform_fit()
snap.write_output()

# Look at the coefficients.
# NOTE: Fit gets sent to first subrank of each comm group internally.
#       Fit will be `None` for all other ranks.
print(f"rank {rank} fit: {snap.solver.fit}")
#print(snap2.solver.fit)

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    



