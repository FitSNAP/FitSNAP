"""
Python script demonstrating a function that takes in an ASE object for a single configuration of
atoms, and then calls FitSNAP on that configuration to calculate fitting data. We then loop over
separate configurations of atoms to extract the fitting data.

We supply a FitSNAP input script which has settings for bispectrum components, reference potential,
and other settings used for fitting.

The SOLVER, GROUPS, EXTRAS, and some other sections in the input script therefore become pointless,
 since we will overwrite the snap.data object to have data associated with a single config from an
ASE atoms object.

Usage:

    python example.py --fitsnap_in Ta-example.in
"""

import numpy as np
from mpi4py import MPI
import argparse
import ase.io
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz
import itertools

def calc_fitting_data(atoms, pt):
    """
    Function to calculate fitting data from FitSNAP.
    Input: ASE atoms object for a single configuration of atoms.
    """

    # make a data dictionary for this config

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'Displaced_BCC'
    data['File'] = None
    data['Stress'] = atoms.get_stress(voigt=False)
    data['Positions'] = atoms.get_positions()
    data['Energy'] = atoms.get_total_energy()
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = atoms.get_forces()
    data['QMLattice'] = atoms.cell[:]
    data['test_bool'] = 0
    data['Lattice'] = atoms.cell[:]
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = 1.0
    data['fweight'] = 1.0
    data['vweight'] = 1.0

    # data must be a list of dictionaries

    data = [data]

    pt.create_shared_array('number_of_atoms', 1, tm=config.sections["SOLVER"].true_multinode)
    pt.shared_arrays["number_of_atoms"].array = np.array([len(atoms)])

    # calculate A matrix for the list of configs in data: 

    snap.data = data
    snap.calculator.shared_index=0
    snap.calculator.distributed_index=0 
    snap.process_configs()

    # return the A matrix for this config
    # we can also return other quantities (reference potential, etc.) associated with fitting

    return pt.shared_arrays['a'].array 

# parse command line args

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
args = parser.parse_args()
print("FitSNAP input script:")
print(args.fitsnap_in)

comm = MPI.COMM_WORLD

# import parallel tools and create pt object

from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm=comm)

# config class reads the input settings

from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])

# create a fitsnap object

from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()

# tell ParallelTool not to create SharedArrays, optional depending on your usage of MPI during fits.
#pt.create_shared_bool = False
# tell ParallelTools not to check for existing fitsnap objects
#pt.check_fitsnap_exist = False

# tell FitSNAP not to delete the data object after processing configs

snap.delete_data = False

# read configs and make a single ASE atoms object 

frames = ase.io.read("Displaced_BCC.xyz", ":")
#atoms = frames[0]

# calculate fitting data using this ASE atoms object

del snap
del config
del pt

for atoms in frames:

    pt = ParallelTools(comm=comm)
    config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
    snap = FitSnap()
    
    fitting_data = calc_fitting_data(atoms, pt)
    print(fitting_data)

    del snap    
    del config
    del pt



