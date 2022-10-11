"""
Test finite difference in forces with custom pairwise network model.
Biggest improvement here was adding a cutoff function to each pairwise interaction. Cutoff functions 
on the descriptors might not be enough.
The cutoff function is necessary because LAMMPS builds a different neigh list in certain scenarios. 
E.g. if rc=4.2, and the largest atom or box is at 4.2, the +/- 0.0001 will result in a different 
number of neighbors for each case according to LAMMPS. You can see this with `Volume_BCC` 
m=0, i=1, a=0, cutoff=4.2

To fix this, we use a cutoff function on each Eij; that way if more neighbors get added, they still 
have a very small contribution near the boundaries.

Alternatively, some people use more strict envelope functions on the descriptors. I don't 
think this is a great idea for NNs because we have bias parameters that can give a contribution 
by simply having an atom move in and out of a neighbor list, regardless of being close at the 
boundary.
"""

import sys
from pathlib import Path
import os
import importlib.util
import numpy as np
from mpi4py import MPI
import argparse
import random
import torch

from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 18})
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
"""
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
"""


this_path = Path(__file__).parent.resolve()
parent_path = Path(__file__).parent.resolve().parent
#example_path = parent_path / 'examples'
ta_example_file = parent_path / 'Ta_pytorch_custom_NN' / 'Ta-example.in'
print(ta_example_file)

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
    fitsnap_in = "Ta-example.in" #ta_example_file.as_posix()
    config = Config(arguments_lst = [fitsnap_in, "--overwrite"])
    #config.sections['BISPECTRUM'].switchflag = 1 # required for smooth finite difference
    config.sections['NETWORK'].manual_seed_flag = 1
    config.sections['NETWORK'].dtype = torch.float64
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
    (energies_model, forces_model) = snap.solver.evaluate_configs(option=1, evaluate_all=True, standardize_bool=True)

    print(f"Length of data: {len(snap.data)}")

    # chose a random config to test against

    #random_indx = random.randint(0, len(snap.data)-1)
    random_indx = 0
    start_indx = 0

    errors = []
    for m in range(random_indx,random_indx+1):
    #for m in range(start_indx, len(snap.data)):
    #for m in range(3,4):
        for i in range(0,snap.data[m]['NumAtoms']):
        #for i in range(1,2):
              for a in range(0,3):
              #for a in range(0,1):
                  natoms = snap.data[m]['NumAtoms']

                  # calculate model energy with +h (energy1)

                  snap.data[m]['Positions'][i,a] += h
                  #print(f"position: {snap.data[m]['Positions'][i,a]}")
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies1, forces1) = snap.solver.evaluate_configs(config_index=m, option=1, standardize_bool=False)

                  # calculate model energy with -h (energy2)

                  snap.data[m]['Positions'][i,a] -= 2.*h
                  #print(f"position: {snap.data[m]['Positions'][i,a]}")
                  snap.calculator.distributed_index = 0
                  snap.calculator.shared_index = 0
                  snap.calculator.shared_index_b = 0
                  snap.calculator.shared_index_c = 0
                  snap.calculator.shared_index_dgrad = 0
                  snap.process_configs()
                  snap.solver.create_datasets()
                  (energies2, forces2) = snap.solver.evaluate_configs(config_index=m, option=1, standardize_bool=False)

                  # calculate and compare finite difference force

                  force_fd = -1.0*(energies1[0] - energies2[0])/(2.*h)
                  force_fd = force_fd.item()
                  force_model = forces_model[m][i][a].item()
                  error = force_model - force_fd

                  print(snap.data[m]['File'])
                  print(f"m i a f_fd f_model: {m} {i} {a} {force_fd} {force_model}")
                  #assert(False)
                  if (abs(error) > 1e-1):
                      assert(False)
                  errors.append(error)

                  # return position back to normal

                  snap.data[m]['Positions'][i,a] += h

    percent_errors = np.array(errors)
    mean_err = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))

    print(f"mean max: {mean_err} {max_err}")
    # mean and max percent error should be < 0.001 %
    # max percent error should be < 0.1 %

    errors = np.abs(errors)

    #print(errors)

    
    hist, bins = np.histogram(errors, bins=10)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(errors, bins=logbins)
    plt.xscale('log')
    plt.xlim((1e-12,1e4))
    plt.xlabel(r'Absolute difference (eV/$\AA$)')
    plt.ylabel("Distribution")
    plt.yticks([])
    plt.savefig("force_check.png", dpi=500)
    

    assert(mean_err < 0.001 and max_err < 0.1)

    del pt
    del config
    del snap.data
    del snap

test_fd_single_elem()