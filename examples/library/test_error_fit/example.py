"""
Python script for performing a fit and immediately calculating test errors after the fit.

Test errors are reported for MAE energy (eV/atom) and MAE force (eV/Angstrom), if using LAMMPS 
metal units.

Usage:

    python example.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example-nodump.in
"""

import numpy as np
from mpi4py import MPI
import argparse
import gc

def calc_mae_energy(df):

    preds = snap.solver.df.loc[:,"preds"]
    truths = snap.solver.df.loc[:, "truths"]
    row_type =  snap.solver.df.loc[:, "Row_Type"]

    # use Row_Type rows to count number of atoms per config
    # see if this number matches that extracted from pt.shared_arrays

    nconfigs = row_type.tolist().count("Energy")
    natoms_per_config = np.zeros(nconfigs).astype(int)

    config_indx = -1
    for element in row_type.tolist():
        if element=="Energy":
            config_indx+=1
        elif element=="Force":
            natoms_per_config[config_indx]+=1
        else:
            pass 

        assert (config_indx>=0)

    natoms_per_config = natoms_per_config/3
    natoms_per_config = natoms_per_config.astype(int)
    #print(natoms_per_config)

    assert ( (natoms_per_config == pt.shared_arrays["number_of_atoms"].array).all() )

    row_indices = row_type[:] == "Energy"
    row_indices = row_indices.tolist()

    test_bool = snap.solver.df.loc[:, "Testing"].tolist()
    test_row_indices = [row_indices and test_bool for row_indices, test_bool in zip(row_indices, test_bool)]
    
    test_truths = np.array(truths[test_row_indices])
    num_test_components = np.shape(test_truths)[0]

    test_preds = np.array(preds[test_row_indices])

    # extract number of atoms for test configs
    # need to know which configs are testing, and which are training, so we extract correct natoms

    test_configs = []
    indx=0
    count=0
    for element in row_type.tolist():
        if (element=="Energy" and test_row_indices[indx]):
            count+=1
            test_configs.append(True)
        if (element=="Energy" and not test_row_indices[indx]):
            test_configs.append(False)
        indx+=1
    
    assert(len(test_configs) == np.shape(natoms_per_config)[0]) 
    assert((sum(test_row_indices) == count) and (sum(test_configs) == count) )

    natoms_test = natoms_per_config[test_configs]
    
    diff = np.abs(test_preds - test_truths)
    diff_per_atom = diff/natoms_test
    mae = np.mean(diff_per_atom)

    return mae

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
snap.delete_data = False

# scrape configs to create the snap data list

snap.scraper.scrape_groups()
snap.scraper.divvy_up_configs()
snap.data = snap.scraper.scrape_configs()

# process configs
# set indices to zero for populating new data array

snap.process_configs()
 
# perform a fit and gather dataframe with snap.solver.error_analysis()
snap.solver.perform_fit()
snap.solver.fit_gather()
snap.solver.error_analysis()

# calculate energy error on the test set

etest_mae = calc_mae_energy(snap.solver.df)

# calculate force error on the test set

ftest_mae = calc_mae_force(snap.solver.df)

print("-----")
print(f"Energy MAE (eV/atom): {etest_mae}")
print(f"Force MAE (eV/Angstrom): {ftest_mae}")
    



