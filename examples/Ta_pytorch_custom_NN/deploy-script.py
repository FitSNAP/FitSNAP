import numpy as np
import pickle
import torch
from pathlib import Path
from fitsnap3lib.tools.write_unified import MLIAPInterface

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
config.sections['NETWORK'].save_state_input = "Ta_Pytorch.pt"
# Create a fitsnap object.
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

model = snap.solver.model
model.dtype = torch.float64
model.eval()

unified = MLIAPInterface(model, ["Ta"], model_device="cpu")
torch.save(unified, "fitsnap_unified.pt")
