"""
Python script for calculating errors on a test set using FitSNAP library.
Before using this script,
1) Declare your pair style in the list below.
2) Read the README

To use this particular example script for the Ta example:
1) Run FitSNAP on the Ta_Linear_JCP2014 example without training on some groups (e.g. the elastic 
groups), by commenting them out in the GROUPS section. This creates a snapparam and snapcoeff file,
that we will use here.
2) Run this example like:

python example2.py --fitsnap_in ../../Ta_Linear_JCP2014/Ta-example.in --test_dir ../../Ta_Linear_JCP2014/Test_Set_Example/
"""
# set pair style commands

pairstyle = ["pair_style hybrid/overlay zbl 4.0 4.8 snap",
             "pair_coeff * * zbl 73 73",
             "pair_coeff * * snap ../Ta_Linear_JCP2014/Ta_pot.snapcoeff ../Ta_Linear_JCP2014/Ta_pot.snapparam Ta"]

import numpy as np
#from mpi4py import MPI # maybe we can parallelize later for huge test sets... no need now
import argparse
import os
from matplotlib import pyplot as plt
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def calc_mae_energy(arr1, arr2):
    abs_diff = np.abs(arr1-arr2)
    mae = np.mean(abs_diff)
    return mae

def calc_mae_force(arr1, arr2):
    diff = arr1-arr2
    norm = np.linalg.norm(diff,axis=1)
    mae = np.mean(norm)
    return mae

# parse command line args

parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
parser.add_argument("--test_dir", help="Test set directory", default=None)
args = parser.parse_args()
print("FitSNAP input script:")
print(args.fitsnap_in)
print("Test directory:")
print(args.test_dir)

#comm = MPI.COMM_WORLD
comm = None

# import parallel tools and create pt object
# this is the backbone of FitSNAP

from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm=comm)

# input class (Config) must be imported after pt

from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
config.sections["PATH"].datapath = args.test_dir
keylist = os.listdir(config.sections["PATH"].datapath)
test_dict = {}
for key in keylist:
    # need to set train_size to 1.0 else FitSNAP will not actually store these configs
    # it is unintuitive but okay for now 
    test_dict[key] = {'training_size': 1.0, 'testing_size': 0.0}
config.sections["GROUPS"].group_table = test_dict

# create a fitsnap object

from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# scrape configs

print(f"Using {config.sections['SCRAPER'].scraper} scraper, change this variable if desired.")
snap.scrape_configs()

from fitsnap3lib.calculators.lammps_snap import LammpsSnap, _extract_compute_np
from fitsnap3lib.calculators.lammps_snap import _extract_compute_np
calc = LammpsSnap(name='LAMMPSSNAP')
calc.shared_index = snap.calculator.shared_index

print(f"Testing on {len(snap.data)} configs.")

energies_all = []
forces_all = []
energies_test_all = []
forces_test_all = []
for i, configuration in enumerate(snap.data):
    #print(i)
    calc._data = configuration
    calc._i = i
    calc._initialize_lammps() # starts a LAMMPS instance
    # set atom style, box, and make atoms
    # this function also clears the previous lammps settings and run
    calc._set_structure()
    # set neighlist
    calc._set_neighbor_list()
    # set your desired pair style
    for pair_command in pairstyle:
        calc._lmp.command(pair_command)
    # run lammps to calculate forces and energies (add a compute for stress if desired)
    calc._lmp.command("compute PE all pe")
    calc._run_lammps()
    num_atoms = calc._data["NumAtoms"]
    lmp_atom_ids = calc._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
    # get forces and energies from lammps
    forces = calc._lmp.numpy.extract_atom("f") # Nx3 array
    energy = calc._lmp.numpy.extract_compute("PE",0,0)
    # get forces and energies from test set
    energy_test = calc._data["Energy"] 
    forces_test = calc._data["Forces"] # Nx3 array
    # append forces and energies to total list
    forces_all.append(forces.tolist())
    energies_all.append(energy)
    forces_test_all.append(forces_test.tolist())
    energies_test_all.append(energy_test)

# prepare data for error calculation

energies_all = np.array(energies_all)
energies_test_all = np.array(energies_test_all)
forces_all = np.concatenate(forces_all,axis=0)
forces_test_all = np.concatenate(forces_test_all, axis=0)
assert ( (np.shape(energies_all) == np.shape(energies_test_all)) )
assert ( (np.shape(forces_all) == np.shape(forces_test_all)) )

# calculate errors

mae_energy = calc_mae_energy(energies_all,energies_test_all)
mae_force = calc_mae_force(forces_all, forces_test_all)
print(f"Units: {config.sections['REFERENCE'].units}")
print(f"Energy MAE: {mae_energy}")
print(f"Force MAE: {mae_force}")

# plot test force vs. model force

import logging # this get's activated in io/outputs/output.py
               # so we need to deactivate it
logging.getLogger('matplotlib.font_manager').disabled = True
min_test_force = np.min(np.abs(forces_test_all.flatten()))
max_test_force = np.max(np.abs(forces_test_all.flatten()))
lims = [min_test_force, max_test_force]
plt.plot(forces_all.flatten(), forces_test_all.flatten(), 'ro', markersize=1)
#plt.plot(dat_val[:,0], dat_val[:,1], 'ro', markersize=2)
plt.plot(lims, lims, 'k-')
plt.legend(["Test Set", "Ideal"])
plt.xlabel("Model force component (eV/A)")
plt.ylabel("Target force component (eV/A)")
plt.xlim(lims[0], lims[1])
plt.ylim(lims[0], lims[1])
plt.savefig("force_comparison.png", dpi=500)
plt.clf()

# plot distribution of force component errors

max_test_force = np.max(np.abs(forces_test_all.flatten()))
lims = [max_test_force, 1.0]
abs_diff = np.abs(forces_all.flatten() - forces_test_all.flatten())
max_abs_diff = np.max(abs_diff)
plt.plot(np.abs(forces_test_all.flatten()), abs_diff, 'ro', markersize=1)
plt.plot([0,max_test_force], [mae_force, mae_force], 'k-', markersize=2)
#plt.plot(lims, lims, 'k-')
plt.legend(["Test Errors", "MAE"])
plt.xlabel("Test set force component (eV/A)")
plt.ylabel("Absolute error (eV/A)")
plt.xlim(0., max_test_force)
plt.ylim(0., max_abs_diff)
plt.savefig("force_comparison_distribution.png", dpi=500)
