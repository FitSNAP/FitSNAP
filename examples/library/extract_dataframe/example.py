"""
Python script for performing a fit using the FitSNAP library, and extracting a dataframe of fitting
quantities..

We do this by creating a FitSNAP object and then calling the functions (see fitsnap3/__main__.py):

snap.scrape_configs()
snap.process_configs()
snap.perform_fit()

After performing the fit, we extract useful info like fitting data and errors, which are contained
in snap.solver.dataframe.

Usage:

Simply run the script and supply a command line arg for the FitSNAP input file path, for example:

    python example.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example-nodump.in
"""

import numpy as np
from mpi4py import MPI
import argparse

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
pt = ParallelTools(comm=comm)
# Config class reads the input
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
# create a fitsnap object
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# scrape configs, process configs, and perform the fit

snap.scrape_configs()
snap.process_configs()
#snap.solver.perform_fit()
snap.perform_fit()

# extract meaningful data from FitSNAP, like the dataframe

print(snap.solver.df)
