"""
ASE scraper is meant to be disconnected from others, and therefore a collection of functions for now.
This is by design since most use cases of ASE desire more flexibility; simply import the functions.
"""

import numpy as np

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

    s.data = [collate_data(atoms) for atoms in frames]

def get_apre(cell):
    """
    Calculate transformed ASE cell for LAMMPS calculations. Thank you Jan Janssen!

    Args:
        cell: ASE atoms cell.

    Returns transformed cell as np.array which is suitable for LAMMPS.
    """
    a, b, c = cell
    an, bn, cn = [np.linalg.norm(v) for v in cell]

    alpha = np.arccos(np.dot(b, c) / (bn * cn))
    beta = np.arccos(np.dot(a, c) / (an * cn))
    gamma = np.arccos(np.dot(a, b) / (an * bn))

    xhi = an
    xyp = np.cos(gamma) * bn
    yhi = np.sin(gamma) * bn
    xzp = np.cos(beta) * cn
    yzp = (bn * cn * np.cos(alpha) - xyp * xzp) / yhi
    zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

    return np.array(((xhi, 0, 0), (xyp, yhi, 0), (xzp, yzp, zhi)))

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