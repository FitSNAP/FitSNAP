"""
Show how to scrape using ASE and then calculate descriptors in FitSNAP.

Usage:

    python example1.py

NOTE: Running in parallel will not loop over configurations in parallel; we need to make a separate 
      list of data dictionaries for each MPI process to do that. See next examples.
"""

import numpy as np
from ase.io import read
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

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
fs = FitSnap(settings, arglist=["--overwrite"])

print("Reading frames")
frames = read("../../Ta_XYZ/XYZ/Displaced_BCC.xyz", ":")[:3]

# Scrape ASE frames into fitsnap data structures. 
data = ase_scraper(frames)

# Loop over configurations and calculate fitting arrays for each separately.
for i, configuration in enumerate(data):
    print(i)
    a,b,w = fs.calculator.process_single(configuration)
    print(np.shape(a))