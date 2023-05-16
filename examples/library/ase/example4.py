"""
Show how to scrape using ASE and then calculate ACE descriptors in FitSNAP.

Usage:

    python example1.py
"""

import numpy as np
from ase.io import read
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

data = \
{
"ACE":
    {
    "numTypes": 2,
    "rcutfac": [5.790, 5.007, 5.007, 4.224],
    "lambda": [1.737, 1.502, 1.502, 1.267],
    "rcinner": [1.705, 1.403, 1.403, 1.100],
    "drcinner": [0.01, 0.01, 0.01, 0.01],
    "ranks": "1 2 3 4",
    "lmax":  "1 2 2 1",
    "nmax":  "22 3 2 1",
    "mumax": "2",
    "lmin": "0 0 1 1",
    "nmaxbase": "22",
    "type": "In P",
    "bzeroflag": 0,
    "bikflag": 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSPACE",
    "energy": 1, # Calculate energy descriptors
    "force": 1,  # Calculate force descriptors
    "stress": 0,  # Calculate virial descriptors
    "per_atom_energy": 1
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
snap = FitSnap(data, arglist=["--overwrite"])

print("Reading frames")
frames = read("../../", ":")[:3]

# Scrape ASE frames into fitsnap data structures.
ase_scraper(snap, frames)

# Loop over configurations and extract fitting arrays.
for i, configuration in enumerate(snap.data):
    snap.pt.single_print(i)
    a,b,w = snap.calculator.process_single(configuration)

    print(np.shape(a))