"""
Show how to scrape using ASE and then calculate descriptors in FitSNAP.
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz

def divvy_up_frames()

def calc_fitting_data(atoms):
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

    #data = [data]

    """
    pt.create_shared_array('number_of_atoms', 1, tm=config.sections["SOLVER"].true_multinode)
    pt.shared_arrays["number_of_atoms"].array = np.array([len(atoms)])
    """

    # calculate A matrix for the list of configs in data: 
    """
    snap.data = data
    snap.calculator.shared_index=0
    snap.calculator.distributed_index=0 
    snap.process_configs()
    """

    # return the A matrix for this config
    # we can also return other quantities (reference potential, etc.) associated with fitting

    #return pt.shared_arrays['a'].array 

    return data

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"main script comm: {comm}")

# Create an input dictionary containing settings.

data = \
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
    "dataPath": "../../../Ta_Linear_JCP2014/JSON"
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
    "Displaced_A15" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
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

print("Making instance")
snap = FitSnap(data, comm=comm, arglist=["--overwrite"])

print("Reading frames")
frames = read("../../Ta_XYZ/XYZ/Displaced_BCC.xyz", ":")[:3]

print("Looping over frames")
data = []
for atoms in frames:
    
    fitting_data = calc_fitting_data(atoms)
    data.append(fitting_data)

print(len(data))

# Now we should be able to:
# (1) inject this data into a fit instance
# (2) which then allocates necessary scraper shared arrays
# (3) then immediately process configs.

snap.delete_data = False
snap.scrape_configs()


