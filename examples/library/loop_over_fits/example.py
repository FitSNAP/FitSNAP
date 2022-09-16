"""
Python script for performing many fits using the FitSNAP library.
Here we loop over many FitSNAP fits and change the weights and descriptor hyperparams each time
with the change_weights and change_descriptor_hyperparams functions.
After changing the hyperparams, we process the configs which creates a new A matrix of descriptors.
Then we perform a fit.
This procedure happens in a loop and is a good foundation for implementing hyperparam optimization.

Usage:

    python example.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example-nodump.in
"""

import numpy as np
from mpi4py import MPI
import argparse
import gc

def calc_mae_force(df):

    preds = snap.solver.df.loc[:,"preds"]
    truths = snap.solver.df.loc[:, "truths"]
    row_type =  snap.solver.df.loc[:, "Row_Type"]

    force_row_indices = row_type[:] == "Force"
    force_row_indices = force_row_indices.tolist()

    testing_bool = snap.solver.df.loc[:, "Testing"].tolist()
    # use list comprehension to extract row indices that are both force and testing
    testing_force_row_indices = [force_row_indices and testing_bool for force_row_indices, testing_bool in zip(force_row_indices, testing_bool)]

    testing_force_truths = np.array(truths[testing_force_row_indices])
    number_of_testing_force_components = np.shape(testing_force_truths)[0]
    testing_force_truths = np.reshape(testing_force_truths, (int(number_of_testing_force_components/3), 3))

    testing_force_preds = np.array(preds[testing_force_row_indices])
    assert(np.shape(testing_force_preds)[0] == number_of_testing_force_components)
    natoms_test = int(number_of_testing_force_components/3)
    testing_force_preds = np.reshape(testing_force_preds, (natoms_test, 3))

    diff = testing_force_preds - testing_force_truths
    norm = np.linalg.norm(diff, axis=1)
    mae = np.mean(norm)

    return mae 

def change_descriptor_hyperparams(config):
    # twojmax, wj, and radelem are lists of chars
    config.sections['BISPECTRUM'].twojmax = ['6']
    config.sections['BISPECTRUM'].wj = ['1.0']
    config.sections['BISPECTRUM'].radelem = ['0.5']
    # rcutfac and rfac0 are doubles
    config.sections['BISPECTRUM'].rcutfac = 4.67637
    config.sections['BISPECTRUM'].rfac0 = 0.99363
    # after changing twojmax, need to generate_b_list to adjust all other variables
    # maybe other hyperparams or descriptors may need other postprocessing functions like this
    config.sections['BISPECTRUM']._generate_b_list()
    return config

def change_weights(config, data):
    # need to find out how many groups there are
    ngroups = len(config.sections['GROUPS'].group_table)
    nweights = 0
    # loop through all group weights in the group_table and change the value
    for key in config.sections['GROUPS'].group_table:
        #print(key)
        for subkey in config.sections['GROUPS'].group_table[key]:
            #print(subkey)
            if ("weight" in subkey):
                nweights += 1
                # change the weight
                config.sections['GROUPS'].group_table[key][subkey] = np.random.rand(1)[0]

    # loop through all configs and set a new weight based on the group table

    for i, configuration in enumerate(data):
        group_name = configuration['Group']
        new_weight = config.sections['GROUPS'].group_table[group_name]
        for key in config.sections['GROUPS'].group_table[group_name]:
            if ("weight" in key):
                # set new weight 
                configuration[key] = config.sections['GROUPS'].group_table[group_name][key]

    return(config, data)

# parse command line args

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
args = parser.parse_args()
print("FitSNAP input script:")
print(args.fitsnap_in)

comm = MPI.COMM_WORLD

# import parallel tools and create pt object
# this is the backbone of FitSNAP
from fitsnap3lib.parallel_tools import ParallelTools
#pt = ParallelTools(comm=comm)
pt = ParallelTools()
print("----- main script")
#print(pt)
# Config class reads the input
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
# create a fitsnap object
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# tell ParallelTool not to create SharedArrays
pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
pt.check_fitsnap_exist = False
# tell FitSNAP not to delete the data object after processing configs
#snap.delete_data = False

# scrape configs to create the snap data list
# we need only do this once, to allocate a data matrix for the configs
# although snap.data will be changed later as we change/optimize the weights
# snap.data contains a list of dictionaries for all configs, with keys for each config 

#snap.scraper.scrape_groups()
#snap.scraper.divvy_up_configs()
#snap.data = snap.scraper.scrape_configs()

del config
del snap

ngenerations = 100
for g in range(0,ngenerations):

    print(f"{g}")

    pt = ParallelTools()
    config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
    snap = FitSnap()

    # tell ParallelTool not to create SharedArrays
    pt.create_shared_bool = False
    # tell ParallelTools not to check for existing fitsnap objects
    pt.check_fitsnap_exist = False
    # tell FitSNAP not to delete the data object after processing configs
    snap.delete_data = True
    snap.scraper.scrape_groups()
    snap.scraper.divvy_up_configs()
    snap.data = snap.scraper.scrape_configs()

    # change the bispectrum hyperparams

    config = change_descriptor_hyperparams(config)

    # change weight hyperparams

    (config, snap.data) = change_weights(config, snap.data)
    
    # process configs with new hyperparams
    # set indices to zero for populating new data array

    snap.calculator.shared_index=0
    snap.calculator.distributed_index=0 
    snap.process_configs()
     
    # perform a fit and gather dataframe with snap.solver.error_analysis()
    #snap.perform_fit()
    #print(pt.fitsnap_dict)
    snap.solver.perform_fit()
    snap.solver.fit_gather()
    # need to empty errors before doing error analysis
    snap.solver.errors = []
    snap.solver.error_analysis()

    # calculate force error on the test set

    ftest_mae = calc_mae_force(snap.solver.df)

    print(f"Generation {g} Force MAE: {ftest_mae}")

    # delete and clean up

    del snap
    del config
    del pt
    



