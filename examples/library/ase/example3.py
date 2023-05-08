"""
Show how to scrape using ASE and then calculate descriptors in FitSNAP.
Here we have a different ASE input than the previous examples; not a list of Atoms objects but 
lists of Atoms objects with positions/cell and other quantities like energy/stress/etc.

Serial use:

    python example3.py

NOTE: When using > 1 process, user must take care that each process has its own list of data.
"""

import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre, create_shared_arrays
import pandas

def ase_scraper(s, frames, energies, forces, stresses):
    """
    Custom function to allocate shared arrays used in Calculator and build the internal list of 
    dictionaries `data` of configuration info. Customized version of `fitsnap3lib.scrapers.ase_funcs`.

    Args:
        s: fitsnap instance.
        frames: list or array of ASE atoms objects.
        energies: array of energies.
        forces: array of forces for all configurations.
        stresses: array of stresses for all configurations.

    Creates a list of data dictionaries `s.data` suitable for fitsnap descriptor calculation.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    create_shared_arrays(s, frames)
    s.data = [collate_data(a, e, f, s) for (a,e,f,s) in zip (frames, energies, forces, stresses)]

def collate_data(atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        energy: energy of a configuration.
        forces: numpy array of forces for a configuration.
        stresses: numpy array of stresses for a configuration.

    Returns a fitsnap data dictionary for a single configuration.
    """

    # make a data dictionary for this config

    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)

    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'Displaced_BCC'
    data['File'] = None
    data['Stress'] = stresses
    data['Positions'] = positions
    data['Energy'] = energy
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = forces
    data['QMLattice'] = cell
    data['test_bool'] = 0
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = 1.0
    data['fweight'] = 1.0/150.0
    data['vweight'] = 0.0

    return data

# Set up your communicator.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create an input dictionary containing settings.

settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 8,
    "rcutfac": 4.812302818,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "W",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1, # Calculate energy descriptors
    "force": 1,  # Calculate force descriptors
    "stress": 0  # Calculate virial descriptors
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "1 1 zbl 74 74"
    }
}

# Create a fitsnap instance.
snap = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Read external data set.
file_name_structures = "filename.h5"
file_name_energies = "filename.hdf"

df_structures = pandas.read_hdf(file_name_structures)
df_structures.sort_index(inplace=True)

df_energies = pandas.read_hdf(file_name_energies)
df_energies.sort_values(by=["index"], inplace=True)

df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]

# Scrape external data into fitsnap data structures.
# TODO: Make this an internal function of `fitsnap3lib.scrapers.ase_funcs`
ase_scraper(snap, df_structures["ASEatoms"].values, df_energies['energy'].values, df_energies["forces"].values, df_energies["stress"].values)
# Now `snap.data` is a list of dictionaries containing configuration/structural info.

# Loop through all configurations and extract fitting data.
for i, configuration in enumerate(snap.data):
    snap.pt.single_print(i)
    # Get A matrix of descriptors, b vector of truths, and weights for this configuration.
    a,b,w = snap.calculator.process_single(configuration, i)

# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
snap.pt.all_barrier()

