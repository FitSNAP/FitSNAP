import torch

# Import parallel tools which is the backbone of FitSNAP.
from fitsnap3lib.parallel_tools import ParallelTools
#pt = ParallelTools(comm=comm)
pt = ParallelTools()
# Don't check for existing fitsnap objects since we may overwrite things.
pt.check_fitsnap_exist = False
from fitsnap3lib.io.input import Config
fitsnap_in = "Ta-example.in"
#fitsnap_in = fitsnap_in.as_posix()
config = Config(arguments_lst = [fitsnap_in, "--overwrite"])
config.sections['PYTORCH'].manual_seed_flag = 1
config.sections['PYTORCH'].dtype = torch.float64
config.sections['PYTORCH'].shuffle_flag = False
config.sections['PYTORCH'].save_state_input = "Ta_Pytorch.pt"
# Only perform calculations on certain structures.
config.sections['GROUPS'].group_table = {'Displaced_BCC': \
    {'training_size': 1.0, \
    'testing_size': 0.0, \
    'eweight': 100.0, \
    'fweight': 1.0, \
    'vweight': 1e-08}}
# Create a fitsnap object.
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# Get config positions.
snap.scrape_configs()
data0 = snap.data
# Don't delete the data since we'll use it many times with finite difference.
snap.delete_data = False 

# Calculate descriptors and prepare data for evaluating.
snap.process_configs()
pt.all_barrier()
# For NNs, create_datasets() stores training data in Configuration objects.
snap.solver.create_datasets()

# Create a list of model energies and forces.
# energies[m] is energy of configuration m.
# forces[m] are forces (torch tensor) of configuration m
(energies, forces) = snap.solver.evaluate_configs(option=1, standardize_bool=False, dtype=torch.float64)

# At this point, we have all we need to calculate custom errors on a test set.
# energies : model energies
# forces : model forces
# snap.data : dictionary of data used by FitSNAP
# snap.solver.configs : list of Configuration objects for better organization of training data.

# These library functions make use of this data to calculate and write errors:

snap.solver.error_analysis()
snap.write_output()