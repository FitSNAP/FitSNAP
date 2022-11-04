"""
Python script for performing a fit and immediately calculating test errors after the fit.

Test errors are reported for MAE energy (eV/atom) and MAE force (eV/Angstrom), if using LAMMPS 
metal units.

Usage:

    python example.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example-nodump.in
"""

import numpy as np
from mpi4py import MPI
import argparse
import gc

# parse command line args

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
args = parser.parse_args()
print("FitSNAP input script:")
print(args.fitsnap_in)

comm = MPI.COMM_WORLD

# import parallel tools and create pt object
# this is the backbone of FitSNAP
from fitsnap3lib.parallel_tools import ParallelTools
#pt = ParallelTools(comm=comm)
pt = ParallelTools(comm=comm)
# Config class reads the input
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
# create a fitsnap object
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()
# import other necessaries to run basic example
from fitsnap3lib.io.output import output
from fitsnap3lib.initialize import initialize_fitsnap_run

# tell ParallelTool not to create SharedArrays
#pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
#pt.check_fitsnap_exist = False
# tell FitSNAP not to delete the data object after processing configs
#snap.delete_data = False

snap.scrape_configs()
snap.process_configs()
pt.all_barrier()
snap.perform_fit()
snap.write_output()

print(snap.solver.fit)
    



