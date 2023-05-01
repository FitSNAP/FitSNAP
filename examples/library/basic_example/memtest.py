"""
Test loops over fitsnap functions and verify no memory leaks.

Usage:

    mpirun -np 2 python memtest.py

Output:

    See output.txt
"""

import numpy as np
import os, psutil
import resource
import time
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import gc
import psutil

def rss():
    return psutil.Process().memory_info().rss

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"main script comm: {comm}")

# Create an input dictionary containing settings.

data = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
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

#infile = "../../WBe_PRB2019/WBe-example.in"

snap = FitSnap(data, comm=comm, arglist=["--overwrite", "--screen2file", "screen.txt"])

# tell ParallelTool not to create SharedArrays
#pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
#snap.pt.check_fitsnap_exist = False
# tell FitSNAP not to delete the data object after processing configs
snap.delete_data = False

snap.scrape_configs()

initial_memory = rss()
for i in range(0,10000000):
    """
    # Creating shared array doesn't seem to leak memory beyond seemingly normal caching.
    snap.pt.create_shared_array('test', int(1e9), tm=snap.config.sections["SOLVER"].true_multinode)
    #snap.pt.shared_arrays.pop("test")
    snap.calculator.distributed_index = 0
    snap.calculator.shared_index = 0
    snap.calculator.shared_index_b = 0
    snap.calculator.shared_index_c = 0
    snap.calculator.shared_index_dgrad = 0
    # This results in increasing mem:
    #snap.process_configs()
    #snap.calculator.create_a()
    #for j in range(0,10):
    #    b = j+1
    """


    """
    snap2 = FitSnap(data, comm=comm, arglist=["--overwrite"])
    #snap2.scraper.divvy_up_configs()
    #snap2.pt.all_barrier()
    #snap2.calculator.create_a()
    #snap2.pt.all_barrier()
    del snap2
    """


    #snap.pt.create_shared_array('a', int(1e9), int(3), tm=snap.config.sections["SOLVER"].true_multinode)
    #snap.pt.new_slice_a()

    #snap.pt.create_shared_array('b', a_len, tm=snap.config.sections["SOLVER"].true_multinode)
    #snap.pt.create_shared_array('w', a_len, tm=snap.config.sections["SOLVER"].true_multinode)
    #self.pt.create_shared_array('ref', a_len, tm=self.config.sections["SOLVER"].true_multinode)
    #snap.pt.new_slice_a()
    #"""

    snap.calculator.distributed_index = 0
    snap.calculator.shared_index = 0
    snap.calculator.shared_index_b = 0
    snap.calculator.shared_index_c = 0
    snap.calculator.shared_index_dgrad = 0
    # This results in increasing mem:
    snap.process_configs()

    snap.pt.free()

    if comm.Get_rank() == 0:
        if i % 1 == 0:
            print(f"{i} {rss()-initial_memory}")

    #snap.pt.shared_arrays['a'].win.Free()