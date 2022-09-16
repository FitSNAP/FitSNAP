"""
Scrape all the configs in a certain directory specified by the FitSNAP input script.

Usage:

    python example.py
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

# import parallel tools and create pt object
# this is the backbone of FitSNAP
from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm=comm)
# Config class reads the input
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = ["../../Ta_Linear_JCP2014/Ta-example.in", "--overwrite"])
# create a fitsnap object
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# scrape configs

snap.scrape_configs()

# create the data matrix for these configs
# if you want to input configs manually, see the create_a function for what is needed

snap.calculator.create_a()
print(f"Found {len(snap.data)} configs")
print("First config keys:")
print(snap.data[0].keys())
#print(snap.calculator.shared_index)

# check if stress fitting is being used, for example

print(f"Fitting stress? {config.sections['CALCULATOR'].stress}")

# populate the shared_arrays for fitting

from fitsnap3lib.calculators.lammps_snap import LammpsSnap, _extract_compute_np
from fitsnap3lib.calculators.lammps_snap import _extract_compute_np

calc = LammpsSnap(name='LAMMPSSNAP')
calc.shared_index = snap.calculator.shared_index

# we can either call the built-in _collect_lammps function, or write our own
#print(LammpsSnap._collect_lammps)

for i, configuration in enumerate(snap.data):
    calc._data = configuration
    calc._i = i
    calc._initialize_lammps()
    calc._prepare_lammps()
    calc._run_lammps()
    # if you wanna use your own collect_lammps function, do this:
    #_collect_lammps(self=calc)
    # otherwise, use the built-in FitSNAP one:
    calc._collect_lammps()
    calc._lmp = pt.close_lammps()

print(pt.shared_arrays["a"].array.shape)

