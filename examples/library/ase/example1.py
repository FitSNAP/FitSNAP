"""
Show how to scrape using ASE and then calculate descriptors in FitSNAP.
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

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
    "energy": 1, # Calculate energy descriptors
    "force": 1,  # Calculate force descriptors
    "stress": 1  # Calculate virial descriptors
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "zero 6.0",
    "pair_coeff": "* *"
    }
}

print("Making instance")
snap = FitSnap(data, comm=comm, arglist=["--overwrite"])

print("Reading frames")
frames = read("../../Ta_XYZ/XYZ/Displaced_BCC.xyz", ":")[:3]

# Scrape ASE frames into fitsnap data structures. 
ase_scraper(snap, frames)

# Create fitsnap dictionaries.
snap.calculator.create_dicts(len(snap.data))
# Create `C` and `d` arrays for solving lstsq with transpose trick.
a_width = snap.calculator.get_width()
for i, configuration in enumerate(snap.data):
    snap.pt.single_print(i)
    a,b,w = snap.calculator.process_single(configuration, i)

    print(np.shape(a))

# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
snap.pt.all_barrier()


