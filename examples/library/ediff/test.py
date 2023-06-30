import lammps
from mpi4py import MPI
import numpy as np
from fitsnap3lib.fitsnap import FitSnap
import torch


# Set up your communicator.
comm = MPI.COMM_WORLD

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
    "num_epochs": 100,
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
    "dataPath": "JSON" #"../../Ta_PyTorch_NN/JSON"
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
    "group_sections": "name training_size testing_size eweight fweight sweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "BCC_Diff"      :  "1.0 0.0 1e2  1 0.003",
    "Displaced_A15" :  "0.7 0.3 1e-2 1 0.003",
    "Displaced_BCC" :  "0.7 0.3 1e-2 1 0.003",
    "Displaced_FCC" :  "0.7 0.3 1e-2 1 0.003",
    "Elastic_BCC"   :  "0.7 0.3 1e-2 1 0.003",
    "Elastic_FCC"   :  "0.7 0.3 1e-2 1 0.003",
    "GSF_110"       :  "0.7 0.3 1e-2 1 0.003",
    "GSF_112"       :  "0.7 0.3 1e-2 1 0.003",
    "Liquid"        :  "0.7 0.3 1e-2 1 0.003",
    "Surface"       :  "0.7 0.3 1e-2 1 0.003",
    "Volume_A15"    :  "0.7 0.3 1e-2 1 0.003",
    "Volume_BCC"    :  "0.7 0.3 1e-2 1 0.003",
    "Volume_FCC"    :  "0.7 0.3 1e-2 1 0.003"
    }
}

fs = FitSnap(settings, arglist=["--overwrite"])
fs.scrape_configs()
fs.process_configs()

# Create list of configs.
fs.solver.create_datasets(pt=fs.pt)

configs = fs.solver.configs

print(len(configs))

# Assign certain configs a pair index to another config.
# Say we wanna focus on \Delta E between two specific configs with known filenames.
#ic1 = [(i,c) for i, c in enumerate(configs) if c.filename == "Ta_liquid_1.json"][0]
#ic2 = [(i,c) for i, c in enumerate(configs) if c.filename == "A15_7.json"][0]
ic1 = [(i,c) for i, c in enumerate(configs) if c.filename == "BCC_1.json"][0]
ic2 = [(i,c) for i, c in enumerate(configs) if c.filename == "FCC_1.json"][0]

print(ic1)
print(ic2)

# Use these tuples to assign a pair index to the config.

ic1[1].pair = ic2[0]
i1 = ic1[0]
i2 = ic2[0]
ic1[1].ediff = configs[i1].energy - configs[i2].energy

# This modifies the `configs` list in place.

#print(configs[ic2[0]].pair)
print(configs[ic1[0]].pair)
print(configs[ic1[0]].ediff)


# Now these configs have a pair attribute, which is the index of the pair config we should take.

fs.solver.perform_fit(configs=configs)


emodel_list = []
for idx, c in enumerate(configs):
    (energies_model, forces_model) = fs.solver.evaluate_configs(config_idx=idx, \
                                                            standardize_bool=False, \
                                                            dtype=torch.float64)
    e_pred = energies_model.detach().numpy()/c.natoms # Model per-atom energy.
    emodel_list.append(e_pred)

ediff = emodel_list[i1] - emodel_list[i2]
ediff_target = configs[i1].energy - configs[i2].energy
ediff_mae = abs(ediff - ediff_target)
print(f"ediff target mae: {ediff} {ediff_target} {ediff_mae}")

