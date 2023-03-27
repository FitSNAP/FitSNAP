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

with open(r"configs.pickle", "rb") as file:
    configs = pickle.load(file)

# Import parallel tools and create corresponding object.
from fitsnap3lib.parallel_tools import ParallelTools
#pt = ParallelTools(comm=comm)
pt = ParallelTools()
# don't check for existing fitsnap objects since we'll be overwriting things
pt.check_fitsnap_exist = False
from fitsnap3lib.io.input import Config
# Declare input script and create config object.
# fitsnap_in = ta_example_file.as_posix() # Use posix if file is Path object
fitsnap_in = "Ta-example.in"
config = Config(arguments_lst = [fitsnap_in, "--overwrite"])
# Load pytorch file from a previous fit.
config.sections['PYTORCH'].save_state_input = "Ta_Pytorch.pt"
# Create a fitsnap object.
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# Calculate model energies/forces.

snap.solver.configs = configs
(energies_model, forces_model) = snap.solver.evaluate_configs(config_idx=None, standardize_bool=True)

print(f"{len(energies_model)} configurations")
print(type(forces_model))
print(forces_model[0])
