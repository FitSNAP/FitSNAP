"""
Demonstrate how to manually inject data into the fitsnap list of data dictionaries prior to 
performing descriptor calculations and/or fits. While we provide ASE scrapers, this example will 
use ASE atoms objects to demonstrate this manual injection.

Serial use:

    python example.py

NOTE: When using > 1 process, user must take care that each process has its own list of data.
      Otherwise you will simply calculate the same descriptors on multiple processes.
"""

import numpy as np
import random
from ase.io import read
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre

def collate_data(atoms):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['Group'] = None
    data['File'] = None
    data['Stress'] = atoms.get_stress(voigt=False)
    data['Positions'] = positions
    data['Energy'] = atoms.get_total_energy()
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = atoms.get_forces()
    data['QMLattice'] = cell
    data['test_bool'] = 0
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    # Inject the weights.
    data['eweight'] = 1.0
    data['fweight'] = 1.0
    data['vweight'] = 1.0

    return data

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
"SOLVER":
    {
    "solver": "SVD"
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
    }
}

# Make a fitsnap instance.
snap = FitSnap(data, comm=comm, arglist=["--overwrite"])

frames = read(f"../../Ta_XYZ/XYZ/Displaced_FCC.xyz", ":")

snap.data = [collate_data(atoms) for atoms in frames]

print(f"Found {len(snap.data)} configurations")

# Calculate descriptors for all configurations.
snap.process_configs()

# Perform a fit.
snap.solver.perform_fit()

# Analyze error metrics.
snap.solver.error_analysis()

# Write error metric and LAMMPS files.
snap.output.output(snap.solver.fit, snap.solver.errors)