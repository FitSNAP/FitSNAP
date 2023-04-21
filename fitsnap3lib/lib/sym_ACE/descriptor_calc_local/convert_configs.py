import numpy as np
import json
from ase import Atoms,Atom
from ase.io import read,write

# function that can be used to convert 'Cu_FHIaims-PBE-dataset.json'
# data from drautz 2019 and lysogorski 2021 to ASE atom objects
def convert(d):
    cell = d["cell"]
    pbc = d["pbc"]
    if d["COORDINATES_TYPE"] == "absolute":
        positions = d["_COORDINATES"]
    elif d["COORDINATES_TYPE"] == "relative":
        scaled_pos = np.array(d["_COORDINATES"])
        positions=np.matmul(cell,scaled_pos.T).T

    symbols = d["_OCCUPATION"]
    natoms = d["NUMBER_OF_ATOMS"]

    ats = []
    for i in range(natoms):
        atom = Atom(symbols[i],positions[i])
        ats.append(atom)
    atoms = Atoms(ats)
    if pbc:
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        return atoms
    else:
        return atoms

#function to convert FitSnap JSON structures
def convert_fitsnap(jsn):
    with open(jsn, 'r') as readin:
        d = json.load(readin)

    pbc = True
    cell = d["Dataset"]["Data"][0]["Lattice"]
    if d["Dataset"]["LatticeStyle"]  != "angstrom":
        raise ValueError("only angstrom units have been implemented here.")
    natoms = d["Dataset"]["Data"][0]["NumAtoms"]
    symbols = d["Dataset"]["Data"][0]["AtomTypes"]
    
    positions = d["Dataset"]["Data"][0]["Positions"]

    ats = []
    for i in range(natoms):
        atom = Atom(symbols[i],positions[i])
        ats.append(atom)
    atoms = Atoms(ats)
    if pbc:
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        return atoms
    else:
        return atoms

def geten_snap(d):
    en = d["Dataset"]["Data"][0]["Energy"]
    nats = d["Dataset"]["Data"][0]["NumAtoms"]
    return en,nats

def geten(d):
    en = d["energy_corrected"]
    nats = d["NUMBER_OF_ATOMS"]
    return en,nats

def get_info(d):
    flag = True
    if d["COORDINATES_TYPE"] == "absolute":
        flag = False
    else:
        pass
    return flag

def convert_snapdataset(d):
    cell = d["Dataset"]["Data"][0]["Lattice"]
    #TODO figure out what fitsnap datasets look like for non-periodic structures
    pbc = True
    natoms = d["Dataset"]["Data"][0]["NumAtoms"]
    if d["Dataset"]["LatticeStyle"] =="angstrom":
        positions = d["Dataset"]["Data"][0]["Positions"]
    elif d["Dataset"]["LatticeStyle"] == "relative":
        scaled_pos = np.array(d["Positions"])
        positions=np.matmul(cell,scaled_pos.T).T
    else:
        print ("I don't know the flag for relative positions")

    if d["Dataset"]["AtomTypeStyle"] =="chemicalsymbol":
        symbols = d["Dataset"]["Data"][0]["AtomTypes"]

    ats = []
    for i in range(natoms):
        atom = Atom(symbols[i],positions[i])
        ats.append(atom)
    atoms = Atoms(ats)
    if pbc:
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        return atoms
    else:
        return atoms
