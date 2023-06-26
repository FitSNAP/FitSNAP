"""
Demonstrate how to scrape with ASE using groups, where each group of configurations has quantities 
such as training/testing fractions, and energy/force/stress weights.

Serial use:

    python example.py

NOTE: When using > 1 process, user must take care that each process has its own list of data.
      Otherwise you will simply calculate the same descriptors on multiple processes.
"""

import numpy as np
import random
from ase.io import read
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre
from fitsnap3lib.scrapers.ase_funcs import ase_scraper
from fitsnap3lib.tools.group_tools import make_table

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create an input dictionary containing settings.

settings = \
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
"SOLVER":
    {
    "solver": "SVD"
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
    }
}

# Make a fitsnap instance.
fs = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# When dealing with groups, it is best to use the `group_table`.
# First make a dictionary of settings for each group, which contains:
# - a key called "group_sections" with a list that names the columns of the table.
# - keys of group names where each key contains a list of column data.

group_settings = {
    "group_sections": ["training_size", "testing_size", "eweight", "fweight", "vweight"],
    "Displaced_A15" :  [1.0,    0.0,      100,            1,               1.00E-08],
    "Displaced_BCC" :  [1.0,    0.0,       100,             1,             1.00E-08],
    "Displaced_FCC" :  [1.0,    0.0,       100,             1,             1.00E-08],
    "Elastic_BCC"   :  [1.0,    0.0,     1.00E-08,        1.00E-08,        0.0001],
    "Elastic_FCC"   :  [1.0,    0.0,     1.00E-09,        1.00E-09,        1.00E-09],
    "GSF_110"       :  [1.0,    0.0,      100,             1,              1.00E-08],
    "GSF_112"       :  [1.0,    0.0,      100,             1,              1.00E-08],
    "Liquid"        :  [1.0,    0.0,       4.67E+02,        1,             1.00E-08],
    "Surface"       :  [1.0,    0.0,       100,             1,             1.00E-08],
    "Volume_A15"    :  [1.0,    0.0,      1.00E+00,        1.00E-09,       1.00E-09],
    "Volume_BCC"    :  [1.0,    0.0,      1.00E+00,        1.00E-09,       1.00E-09],
    "Volume_FCC"    :  [1.0,    0.0,      1.00E+00,        1.00E-09,       1.00E-09]
    }

group_table = make_table(group_settings)

# Make ASE frames for each group; do this however you want, we simply read from filenames that share 
# group names here.
for name in group_table:
    frames = read(f"../../Ta_XYZ/XYZ/{name}.xyz", ":")
    group_table[name]["frames"] = frames
    group_table[name]["nconfigs"] = len(frames)

# NOTE: For parallelism, at this point we could distribute frames in each group across procs.

# Inject group data into the fitsnap list of data dictionaries.
data = ase_scraper(group_table)

print(f"Found {len(data)} configurations")

# Calculate descriptors for all configurations.
fs.process_configs(data)

# Perform a fit.
fs.solver.perform_fit()

# Analyze error metrics.
fs.solver.error_analysis()

# Write error metric and LAMMPS files.
fs.output.output(fs.solver.fit, fs.solver.errors)

# Dataframe of detailed errors per group.
print(fs.solver.errors)