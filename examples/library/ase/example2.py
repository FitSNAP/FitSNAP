"""
Demonstrate how to scrape using ASE and then calculate descriptors in FitSNAP *in parallel*.

This example starts with a list of ASE atoms objects in `loaded_frames`. 
Then we manually split the objects over procs; in real application perhaps each proc has its own 
ASE list.

Usage:

    mpirun -np 2 python example2.py
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import ase
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
"MEMORY":
    {
    "override": 0
    }
}

print("Making instance")
snap = FitSnap(data, comm=comm, arglist=["--overwrite"])

# Illustrate how to use the ASE scraper in parallel.
# The ASE frames are loaded on all MPI processes.
# We must split the frames manually, shown with 2 processes below.
# NOTE: If you don't want to do this in parallel, simply feed `frames_all` into the `ase_scraper`.
frames_all = read("../../Ta_XYZ/XYZ/Displaced_BCC.xyz", ":")[:6]
if (rank == 0):
    print(f"Reading frames on rank {rank}")
    frames = frames_all[rank:3]
elif (rank == 1):
    print(f"Reading frames on rank {rank}")
    frames = frames_all[3:]

snap.pt.all_barrier()

# Scrape frames into fitsnap data structures.
ase_scraper(snap, frames)

# Calculate descriptors for all configurations.
snap.process_configs()

# Perform a fit.
snap.solver.perform_fit()

# Analyze error metrics.
snap.solver.error_analysis()

# Write error metric and LAMMPS files.
snap.output.output(snap.solver.fit, snap.solver.errors)

