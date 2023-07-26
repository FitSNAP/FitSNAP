"""
This script loads a pickled list of FitSNAP Configuration objects then inputs 
these into a NN for simply evaluating energies and forces on the data set.

Usage:

    python evaluate_configs.py

NOTE: Change the following settings based on your fit:
      fitsnap_in (str) : Name of fitsnap input script.
      config.sections['PYTORCH'].save_state_input (str) : Name of Pytorch .pt file.
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from fitsnap3lib.fitsnap import FitSnap

with open(r"configs.pickle", "rb") as file:
    configs = pickle.load(file)

# NOTE: Settings dictionary needs to be same as input script.
# TODO: Relieve dependence on using unrelated sections like BISPECTRUM and CALCULATOR for simple NN eval.
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
    "save_state_input": "Ta_Pytorch.pt"
    },
"SOLVER":
    {
    "solver": "PYTORCH"
    }
}

fs = FitSnap(settings, arglist=["--overwrite"])

fs.solver.configs = configs
(energies_model, forces_model) = fs.solver.evaluate_configs(config_idx=None, standardize_bool=True)

# Convert to numpy arrays.
nconfigs = len(energies_model)
for m in range(nconfigs):
    energies_model[m] = energies_model[m].detach().numpy().astype(float)
    forces_model[m] = forces_model[m].detach().numpy()

print(f"{len(energies_model)} configurations")
print(energies_model[0])

# We return total energies, so calculate per-atom energies like:
print(energies_model[0]/configs[0].natoms)
