"""
Use the JSON scraper to collate a list of fitsnap data dictionaries containing fitting info.
Then loop over configurations and extract the fitting arrays (A, b, w) for each.

Serial use:

    python example2.py

Parallel use:

    mpirun -np P python example2.py

NOTE: When using in parallel, the `fitsnap.data` list is distributed over MPI processes, so each 
      process recieves a portion of the data.
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

# Declare an optional communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "/Users/adrohsk/FitSNAP/examples/Ta_Linear_JCP2014/JSON"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "zero 6.0",
    "pair_coeff": "* *"
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08"
    }
}

# Make an instance and allow overwriting of possible output files.
fitsnap = FitSnap(data, arglist=["--overwrite"])

# Scrape JSON files into the `fitsnap.data` list of dictionaries containing fitting info.
fitsnap.scrape_configs()

for i, configuration in enumerate(fitsnap.data):
    # Calculate fitting arrays (A, b, w) for this configuration.
    a,b,w = fitsnap.calculator.process_single(configuration)
    print(f"Rank {rank} file {configuration['File']} A matrix size: {np.shape(a)}")
    print(a)
