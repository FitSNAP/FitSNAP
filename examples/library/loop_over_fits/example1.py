"""
Python script for performing many fits using the FitSNAP library.
Here we loop over many FitSNAP fits and change the weights and descriptor hyperparams each time
with the `change_weights` and `change_descriptor_hyperparams` functions.
After changing these hyperparams, we process the configs which creates a new A matrix of descriptors.
Then we perform a fit.
This procedure happens in a loop and is a good foundation for implementing hyperparam optimization.

Usage:

    python example.py
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

def change_descriptor_hyperparams(s):
    """
    Change descriptor hyperparameters associated with a particular fitsnap config object.
    Args:
        s: A fitsnap instance.
    No return since we modify the state of a fitsnap instance due to Python passing function args 
    by reference.
    """
    # twojmax, wj, and radelem are lists of chars
    s.config.sections['BISPECTRUM'].twojmax = ['6']
    s.config.sections['BISPECTRUM'].wj = ['1.0']
    s.config.sections['BISPECTRUM'].radelem = ['0.5']
    # rcutfac and rfac0 are doubles
    s.config.sections['BISPECTRUM'].rcutfac = 3.67637
    s.config.sections['BISPECTRUM'].rfac0 = 0.99363
    # After changing twojmax, need to generate_b_list to adjust all other variables.
    # Maybe other hyperparams or descriptors may need other postprocessing functions like this.
    s.config.sections['BISPECTRUM']._generate_b_list()

def change_weights(s):
    """
    Change weight hyperparams associated with a fitsnap instance.
    Args:
        s: A fitsnap instance.
    No return since we modify the state of a fitsnap instance due to Python passing function args 
    by reference.
    """
    # need to find out how many groups there are
    ngroups = len(s.config.sections['GROUPS'].group_table)
    nweights = 0
    # loop through all group weights in the group_table and change the value
    for key in s.config.sections['GROUPS'].group_table:
        for subkey in s.config.sections['GROUPS'].group_table[key]:
            if ("weight" in subkey):
                nweights += 1
                # change the weight
                s.config.sections['GROUPS'].group_table[key][subkey] = np.random.rand(1)[0]

    # loop through all configs and set a new weight based on the group table

    for configuration in s.data:
        group_name = configuration['Group']
        for key in s.config.sections['GROUPS'].group_table[group_name]:
            if ("weight" in key):
                # set new weight based on previously changed group table value
                configuration[key] = s.config.sections['GROUPS'].group_table[group_name][key]

    #return(config, data)

# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD

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
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "../../Ta_Linear_JCP2014/JSON"
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 1,
    "dump_truth": 1,
    "dump_weights": 1,
    "dump_dataframe": 1
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "0.7    0.3       100             1               1.00E-08",
    "Displaced_BCC" :  "0.7    0.3       100             1               1.00E-08",
    "Displaced_FCC" :  "0.7    0.3       100             1               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0     1.00E-08        1.00E-08        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.00E-09        1.00E-09        1.00E-09",
    "GSF_110"       :  "1.0    0.0      100             1               1.00E-08",
    "GSF_112"       :  "1.0    0.0      100             1               1.00E-08",
    "Liquid"        :  "1.0    0.0       4.67E+02        1               1.00E-08",
    "Surface"       :  "1.0    0.0       100             1               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Create a FitSnap instance using the communicator and settings:
fitsnap = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configs one time at the beginning.
# This populates the `fitsnap.data` dictionary of fitting info.
fitsnap.scrape_configs()

ngenerations = 100
for g in range(0,ngenerations):

    print(f"{g}")

    # Change descriptor hyperparams of this instance.

    change_descriptor_hyperparams(fitsnap)

    # Change weight hyperparams of this instance.

    change_weights(fitsnap)
    
    # Process configs with new hyperparams.

    fitsnap.process_configs()

    fitsnap.solver.perform_fit()

    fitsnap.solver.error_analysis()

    # Extract desired errors from the `fitsnap.solver.errors` dataframe.
    # E.g. force test MAE is found with the following.

    ftest_mae = fitsnap.solver.errors['mae'][('*ALL', 'Unweighted', 'Testing', 'Force')]

    print(f"Generation {g} Force MAE: {ftest_mae}")
