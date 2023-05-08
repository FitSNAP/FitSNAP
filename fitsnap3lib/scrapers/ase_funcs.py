"""
ASE scraper is meant to be disconnected from others, and therefore a collection of functions for now.
This is by design since most use cases of ASE desire more flexibility; simply import the functions.
"""

import numpy as np
from ase import Atoms,Atom
from ase.io import read,write
from ase.io import extxyz
from mpi4py import MPI

def ase_scraper(s, frames):
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.
    TODO: Let user assign group names for ASE frames with a dictionary of group names, testing 
          bools, weights, etc.

    Args:
        s: fitsnap instance.
        frames: ASE frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    print(frames)

    # TODO: If user doesn't supply a group, just default to some ALL group.
    # NODES SPLIT UP HERE
    # self.configs = self.pt.split_by_node(self.configs)
    """
    self.test_bool = self.pt.split_by_node(self.test_bool)
    groups = self.pt.split_by_node(groups)
    group_list = self.pt.split_by_node(group_list)
    temp_configs = copy(self.configs)
    """
    # Single group for now:
    group_counts = np.zeros((len(frames),), dtype='i')

    # TODO: `self.tests` is a list of filenames associated with test configs.
    #       Could be implemented later to include ASE frames.
    """
    if self.tests is not None:
        self.pt.shared_arrays['configs_per_group'].testing = len(test_list)
    """

    # Reduce length of frames across procs.
    len_frames = np.array([len(frames)])
    len_frames_all = np.array([0])
    s.pt._comm.Allreduce([len_frames, MPI.INT], [len_frames_all, MPI.INT])

    number_of_configs_per_node = int(len_frames_all)
    s.pt.create_shared_array('number_of_atoms', number_of_configs_per_node, dtype='i')
    s.pt.slice_array('number_of_atoms')
    #s.pt.shared_arrays['number_of_atoms'].configs = frames #temp_configs

    # number of dgrad rows serves similar purpose as number of atoms
    
    s.pt.create_shared_array('number_of_dgrad_rows', number_of_configs_per_node, dtype='i')
    s.pt.slice_array('number_of_dgrad_rows')
    s.pt.shared_arrays['number_of_dgrad_rows'].configs = frames #temp_configs

    # number of neighs serves similar purpose as number of atoms for custom calculator
    
    s.pt.create_shared_array('number_of_neighs_scrape', number_of_configs_per_node, dtype='i')
    s.pt.slice_array('number_of_neighs_scrape')

    # PROCS SPLIT UP HERE
    #s.test_bool = s.pt.split_within_node(self.test_bool)
    # Don't distribute frames among procs in the ASE scraper, let's assume each proc as its own 
    # set of frames.
    # s.solver.configs = s.pt.split_within_node(frames)
    #s.solver.configs = frames

    # Set number of atoms in the sliced arrays used in Calculator.

    #for i, frame in enumerate(s.solver.configs):
    for i, frame in enumerate(frames):
        natoms = len(frame)
        print(f"rank {s.pt._rank} natoms {natoms}")
        s.pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms

    # Loop through atoms objects and build the data list.

    #s.data = [collate_data(atoms) for atoms in s.solver.configs]
    s.data = [collate_data(atoms) for atoms in frames]

def collate_data(atoms):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: ASE atoms object for a single configuration of atoms.

    Returns a data dictionary for a single configuration.
    """

    # make a data dictionary for this config

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'ASE' # TODO: Make this customizable for ASE groups.
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

    return data