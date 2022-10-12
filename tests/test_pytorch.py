import pytest
import sys
from pathlib import Path
import os
import importlib.util
import numpy as np
from mpi4py import MPI
import argparse
import random
import torch


this_path = Path(__file__).parent.resolve()
parent_path = Path(__file__).parent.resolve().parent
example_path = parent_path / 'examples'
ta_example_file = example_path / 'Ta_PyTorch_NN' / 'Ta-example.in'
wbe_example_file = example_path / 'WBe_PyTorch_NN' / 'WBe-example.in'

def test_fd_single_elem():
    # TODO: Make equivalent test using MPI

    h = 1e-4 # size of finite difference

    # import parallel tools and create pt object
    from fitsnap3lib.parallel_tools import ParallelTools
    #pt = ParallelTools(comm=comm)
    pt = ParallelTools()
    # don't check for existing fitsnap objects since we'll be overwriting things
    pt.check_fitsnap_exist = False
    from fitsnap3lib.io.input import Config
    #fitsnap_in = "../examples/Ta_Pytorch_NN/Ta-example.in"
    fitsnap_in = ta_example_file.as_posix()
    config = Config(arguments_lst = [fitsnap_in, "--overwrite"])
    config.sections['BISPECTRUM'].switchflag = 1 # required for smooth finite difference
    config.sections['PYTORCH'].manual_seed_flag = 1
    config.sections['PYTORCH'].dtype = torch.float64
    # only perform calculations on displaced BCC structures
    config.sections['GROUPS'].group_table = {'Displaced_BCC': \
        {'training_size': 1.0, \
        'testing_size': 0.0, \
        'eweight': 100.0, \
        'fweight': 1.0, \
        'vweight': 1e-08}}
    # create a fitsnap object
    from fitsnap3lib.fitsnap import FitSnap
    snap = FitSnap()

    # get config positions
    snap.scrape_configs()
    data0 = snap.data
    # don't delete the data since we'll use it many times with finite difference
    snap.delete_data = False 

    # calculate model forces

    snap.process_configs()
    pt.all_barrier()
    snap.solver.create_datasets()
    (energies_model, forces_model) = snap.solver.evaluate_configs(option=1, standardize_bool=True)

    print(f"Length of data: {len(snap.data)}")

    # chose a random config to test against

    random_indx = 0 #random.randint(0, len(snap.data)-1)

    percent_errors = []
    for m in range(random_indx,random_indx+1):
        for i in range(0,snap.data[m]['NumAtoms']):
        #for i in range(0,1):
              for a in range(0,3):
                  natoms = snap.data[m]['NumAtoms']

                  # calculate model energy with +h (energy1)

                  snap.data[m]['Positions'][i,a] += h
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies1, forces1) = snap.solver.evaluate_configs(option=1, standardize_bool=False)

                  # calculate model energy with -h (energy2)

                  snap.data[m]['Positions'][i,a] -= 2.*h
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies2, forces2) = snap.solver.evaluate_configs(option=1, standardize_bool=False)

                  # calculate and compare finite difference force

                  force_fd = -1.0*(energies1[m] - energies2[m])/(2.*h)
                  force_fd = force_fd.item()
                  force_indx = 3*i + a 
                  force_model = forces_model[m][force_indx].item()
                  percent_error = abs(force_model - force_fd) #((force_model - force_fd)/force_model)*100.
                  percent_errors.append(percent_error)

                  # return position back to normal

                  snap.data[m]['Positions'][i,a] += h

    percent_errors = np.array(percent_errors)
    mean_err = np.mean(np.abs(percent_errors))
    max_err = np.max(np.abs(percent_errors))

    # mean and max percent error should be < 0.001 %
    # max percent error should be < 0.1 %

    assert(mean_err < 0.001 and max_err < 0.1)

    del pt
    del config
    del snap.data
    del snap

def test_fd_multi_elem():
    # TODO: Make equivalent test using MPI

    h = 1e-4 # size of finite difference

    # import parallel tools and create pt object
    from fitsnap3lib.parallel_tools import ParallelTools
    #pt = ParallelTools(comm=comm)
    pt = ParallelTools()
    # don't check for existing fitsnap objects since we'll be overwriting things
    pt.check_fitsnap_exist = False
    from fitsnap3lib.io.input import Config
    #fitsnap_in = "../examples/WBe_Pytorch_NN/WBe-example.in"
    fitsnap_in = wbe_example_file.as_posix()
    config = Config(arguments_lst = [fitsnap_in, "--overwrite"])
    config.sections['BISPECTRUM'].switchflag = 1 # required for smooth finite difference
    config.sections['PYTORCH'].manual_seed_flag = 1
    config.sections['PYTORCH'].dtype = torch.float64
    # only perform calculations on a certain group
    config.sections['GROUPS'].group_table = {'DFT_MD_300K': \
        {'training_size': 0.01, \
        'testing_size': 0.0, \
        'eweight': 100.0, \
        'fweight': 1.0, \
        'vweight': 1e-08}}
    # create a fitsnap object
    from fitsnap3lib.fitsnap import FitSnap
    snap = FitSnap()

    # get config positions
    snap.scrape_configs()
    data0 = snap.data
    # don't delete the data since we'll use it many times with finite difference
    snap.delete_data = False 

    # calculate model forces

    snap.process_configs()
    pt.all_barrier()
    snap.solver.create_datasets()
    (energies_model, forces_model) = snap.solver.evaluate_configs(option=1, standardize_bool=True)

    print(f"Length of data: {len(snap.data)}")

    # chose a random config to test against

    random_indx = 0 #random.randint(0, len(snap.data)-1)

    percent_errors = []
    for m in range(random_indx,random_indx+1):
        for i in range(0,snap.data[m]['NumAtoms']):
              for a in range(0,3):
                  natoms = snap.data[m]['NumAtoms']

                  # calculate model energy with +h (energy1)

                  snap.data[m]['Positions'][i,a] += h
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies1, forces1) = snap.solver.evaluate_configs(option=1, standardize_bool=False)

                  # calculate model energy with -h (energy2)

                  snap.data[m]['Positions'][i,a] -= 2.*h
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies2, forces2) = snap.solver.evaluate_configs(option=1, standardize_bool=False)

                  # calculate and compare finite difference force

                  force_fd = -1.0*(energies1[m] - energies2[m])/(2.*h)
                  force_fd = force_fd.item()
                  force_indx = 3*i + a 
                  force_model = forces_model[m][force_indx].item()
                  percent_error = abs(force_model - force_fd) #((force_model - force_fd)/force_model)*100.
                  assert(percent_error < 0.1) # nice assertion to have for all forces
                  percent_errors.append(percent_error)

                  # return position back to normal

                  snap.data[m]['Positions'][i,a] += h

    percent_errors = np.array(percent_errors)
    mean_err = np.mean(np.abs(percent_errors))
    max_err = np.max(np.abs(percent_errors))

    # mean and max percent error should be < 0.001 %
    # max percent error should be < 0.1 %

    assert(mean_err < 0.001 and max_err < 0.1)

    del pt
    del config
    del snap.data
    del snap