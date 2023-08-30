"""
ASE scraper is meant to be disconnected from others, and therefore a collection of functions for now.
This is by design since most use cases of ASE desire more flexibility; simply import the functions.
"""

import numpy as np
from fitsnap3lib.tools.group_tools import assign_validation


def ase_scraper(data, random_test: bool=False) -> list:
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.

    Args:
        data: List of ASE frames or dictionary group table containing frames.
        random_test: Select test data randomly if True, else take last percentage of configs in a group for 
                     reproducibility.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    # Simply collate data from Atoms objects if we have a list of Atoms objects.
    if type(data) == list:
        return [collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    elif type(data) == dict:
        assign_validation(data, random_test=random_test)
        ret = []
        for name in data:
            frames = data[name]["frames"]
            # Extend the fitsnap data list with this group.
            ret.extend([collate_data(atoms, name, data[name], f) for f, atoms in enumerate(frames)])
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

def collate_data(atoms, name: str=None, group_dict: dict=None, f: int=0) -> dict:
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.
        f: Optional index associated with configuration in a group.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['Group'] = name # TODO: Make this customizable for ASE groups.
    data['File'] = f"{name}_{f}"
    data['Positions'] = positions
    data['AtomTypes'] = atoms.get_chemical_symbols()
    if (atoms.calc is None):
        # Just calculating descriptors; assign 0.
        data['Energy'] = 0.0
        data['Forces'] = np.zeros((len(atoms), 3))
        data['Stress'] = np.zeros(6)
    else:
        data['Energy'] = atoms.get_total_energy()
        data['Forces'] = atoms.get_forces()
        data['Stress'] = atoms.get_stress(voigt=False)
    data['NumAtoms'] = len(atoms)
    data['QMLattice'] = cell
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    # Inject the weights and other group quantities.
    if group_dict is not None:
        data['eweight'] = group_dict["eweight"] if "eweight" in group_dict else 1.0
        data['fweight'] = group_dict["fweight"] if "fweight" in group_dict else 1.0
        data['vweight'] = group_dict["vweight"] if "vweight" in group_dict else 1.0
        data['test_bool'] = group_dict['test_bools'][f]
    else:
        data['eweight'] = 1.0
        data['fweight'] = 1.0
        data['vweight'] = 1.0
        data['test_bool'] = 0

    return data