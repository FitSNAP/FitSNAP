"""
ASE scraper is meant to be disconnected from others, and therefore a collection of functions for now.
This is by design since most use cases of ASE desire more flexibility; simply import the functions.
"""

import numpy as np
from fitsnap3lib.tools.group_tools import assign_validation

#def ase_scraper(s, data):
def ase_scraper(data) -> list:
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    # Simply collate data from Atoms objects if we have a list of Atoms objects.
    if type(data) == list:
        #s.data = [collate_data(atoms) for atoms in data]
        return [collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    elif type(data) == dict:
        assign_validation(data)
        #s.data = []
        ret = []
        for name in data:
            frames = data[name]["frames"]
            # Extend the fitsnap data list with this group.
            #s.data.extend([collate_data(atoms, name, data[name]) for atoms in frames])
            ret.extend([collate_data(atoms, name, data[name]) for atoms in frames])
        return ret
    else:
        raise Exception("Argument must be list or dictionary for ASE scraper.")
        

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

def collate_data(atoms, name: str=None, group_dict: dict=None) -> dict:
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['Group'] = name #'ASE' # TODO: Make this customizable for ASE groups.
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
    if group_dict is not None:
        data['eweight'] = group_dict["eweight"] if "eweight" in group_dict else 1.0
        data['fweight'] = group_dict["fweight"] if "fweight" in group_dict else 1.0
        data['vweight'] = group_dict["vweight"] if "vweight" in group_dict else 1.0
    else:
        data['eweight'] = 1.0
        data['fweight'] = 1.0
        data['vweight'] = 1.0

    return data