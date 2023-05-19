"""
Python script for performing many fits using the FitSNAP library.
Here we calculate descriptors once then loop over many fits.
We achieve this by:
1. Calculating descriptors one time using a single instance.
2. Using this data as the argument to solver functions of other instances.
3. Perform NN fits with these instances.

Usage:

    mpirun -np 2 python example.py

If running in parallel, we create a different instance on each proc and run different NN fits on 
each proc.
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize settings in a traditional input file.
# NOTE: These settings will be changed as we loop through configurations.
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
    "bzeroflag": 1,
    "bikflag": 1,
    "dgradflag": 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "per_atom_energy": 1,
    "nonlinear": 1
    },
"PYTORCH":
    {
    "layer_sizes": "num_desc 64 64 1",
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "batch_size": 4, # 363 configs in entire set
    "save_state_output": "Ta_Pytorch.pt"
    },
"SOLVER":
    {
    "solver": "PYTORCH"
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "../../Ta_Linear_JCP2014/JSON"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 6.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight",
    "group_types": "str float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "0.7 0.3 1e-2 1",
    "Displaced_BCC" :  "0.7 0.3 1e-2 1",
    "Displaced_FCC" :  "0.7 0.3 1e-2 1",
    "Elastic_BCC"   :  "0.7 0.3 1e-2 1",
    "Elastic_FCC"   :  "0.7 0.3 1e-2 1",
    "GSF_110"       :  "0.7 0.3 1e-2 1",
    "GSF_112"       :  "0.7 0.3 1e-2 1",
    "Liquid"        :  "0.7 0.3 1e-2 1",
    "Surface"       :  "0.7 0.3 1e-2 1",
    "Volume_A15"    :  "0.7 0.3 1e-2 1",
    "Volume_BCC"    :  "0.7 0.3 1e-2 1",
    "Volume_FCC"    :  "0.7 0.3 1e-2 1"
    }
}

# Create a FitSnap instance using the communicator and settings:
instance1 = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configs one time at the beginning.
# This populates the `fitsnap.data` dictionary of fitting info.
instance1.scrape_configs()
# Process configurations and gather internal lists to all processes, since we will use multiple 
# processes for training multiple NNs.
instance1.process_configs(allgather=True)

# Split the communicator by declaring colors and keys on each proc.
# color: determines which communicator the proc will be in.
# key: determines the rank of the proc in the new communicator.
if rank == 0:
    color = rank
    key = rank
elif rank == 1:
    color = rank
    key = rank

comm_split = MPI.COMM_WORLD.Split(color, key)

rank_split = comm_split.Get_rank()
size_split = comm_split.Get_size()
print(f"rank {rank} color {color} key {key} size_split {size_split} rank_split {rank_split}")

# Declare unique settings for each rank by deepcopying settings and modifying certain fields.
from copy import deepcopy
if rank == 0:
    settings2 = deepcopy(settings)
    settings2["PYTORCH"]["num_epochs"] = 10
elif rank == 1:
    settings2 = deepcopy(settings)
    settings2["PYTORCH"]["num_epochs"] = 20

instance2 = FitSnap(settings2, comm=comm_split, arglist=["--overwrite"])

# Perform the fit with some special arguments:
# - `pt` instance to fit on data from another instance.
# - `outfile` to write fitting progress of this particular rank.
# - `verbose` is False because printing progress of each proc to the screen is messy.
instance2.solver.perform_fit(pt = instance1.pt, outfile = f"rank{rank}_progress.txt", verbose = False)