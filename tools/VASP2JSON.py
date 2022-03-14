#! /usr/bin/env/ python

# This script will parse through a single OUTCAR file from VASP, which may include one or more configurations, and will print out 
# a JSON file(s) that can then be read into fitSNAP.  To run this script, you will need to specify an OUTCAR file, a POSCAR file, and 
# the name of the JSON file that will be output in  the command line  --->  python VASP2JSON.py myOUTCARfile myPOSCARfile myJSONfile 

import sys, os
import json

OUTCAR_file = sys.argv[1]
POSCAR_file = sys.argv[2]
JSON_file = str(sys.argv[3])

# Define a function to take all the data that will be needed and send that to a 
# json.dump command which then takes that data and puts it into json format.  
# The input for this function (data) is a dictionary that holds all the values from 
# the OUTCAR file that will be parsed through (such as atom positions, forces, lattice 
# vectors, etc.) which will be put into all Data with key 'Data'.  allDataHeader holds 
# all of the units as well as 'Data'.  This is then encompassed by myDataSet which then gets 
# fed into the json.dump command.  Most of this is just getting everything into the right format
def write_json(data,jsonfilename):
	jsonfile = open(jsonfilename,'w')
	#print >>jsonfile, "# Header\n"
	#print("# Header", file=jsonfilename)
	allData = [data]
	allDataHeader = {}
	allDataHeader['EnergyStyle'] = "electronvolt"
	allDataHeader['StressStyle'] = "kB"
	allDataHeader['AtomTypeStyle'] = "chemicalsymbol"
	allDataHeader['PositionsStyle'] = "angstrom"
	allDataHeader['ForcesStyle'] = "electronvoltperangstrom"
	allDataHeader['LatticeStyle'] = "angstrom"
	allDataHeader['Data'] = allData

	myDataset = {}

	myDataset['Dataset'] = allDataHeader
	sjson = json.dump(myDataset, jsonfile, indent = 2, sort_keys = True)
	return

# print_num_atoms_per_type takes the POSCAR specified by the user and parses it 
# for the  number of each atom types, which is assumed to be 
# in lines 7 of the POSCAR file (so if that's not the case this will need 
# to be modified in atomTypeLine and numberAtomsLine).  This function then returns 
# a the number of the atom type for each indvidual atoms for each configuration.
# Line 6 is the POSCAR is not actually used by VASP, and the ordering of the 
# atom types is determined by the POTCAR file, and listed in the OUTCAR file.

def print_num_atoms_per_type(myPOSCARFile):
	f1 = open(myPOSCARFile, 'r+')
	lines = f1.readlines()
	numberAtomsLine = lines[6]
	columnsNumberAtom = numberAtomsLine.split()
        num_atoms_per_type = [int(num) for num in columnsNumberAtom]
        return num_atoms_per_type


num_atoms_per_type = print_num_atoms_per_type(POSCAR_file)
num_atom_types = len(num_atoms_per_type)
order_atom_types = []
listAtomTypes = []

# Start parsing through OUTCAR looking for keywords that are assocaited with the 
# different values for the data needed, such as forces or positions

f2 = open(OUTCAR_file, 'r+')
lines = f2.readlines()
outcar_config_number = 1

for i in range(0, len(lines)):
	line = lines[i]
        # Look for the ordering of the atom types 
        # can grab atom labels from VRHFIN lines at top of OUTCAR
        # (These will only show up once for each element at in the OUTCAR)
        if 'VRHFIN' in line:
                        order_atom_types.append(line.split()[1][1:].strip(" :"))  
                        if len(order_atom_types) == num_atom_types:
                                for atom_type in range(0, num_atom_types):
                                        totalAtoms = int(num_atoms_per_type[atom_type])
                                        atomType = str(order_atom_types[atom_type])
                                        for n in range(0,totalAtoms):
                                                listAtomTypes.append(atomType)
	# Look for number of atoms in configuration
	if 'number of ions' in line:
		columns = line.split()
		natoms = int(columns[11])
	# Look for lattice vectors for configuration
	if 'direct lattice vectors' in line:
		line = lines[i+1]
		columns = line.split()
		x1 = float(columns[0])
		x2 = float(columns[1])
		x3 = float(columns[2])

		line = lines[i+2]
		columns = line.split()
		y1 = float(columns[0])
		y2 = float(columns[1])
		y3 = float(columns[2])

		line = lines[i+3]
		columns = line.split()
		z1 = float(columns[0])
		z2 = float(columns[1])
		z3 = float(columns[2])

		lattice_x = [x1, x2, x3]
		lattice_y = [y1, y2, y3]
		lattice_z = [z1, z2, z3]

		all_lattice = [lattice_x, lattice_y, lattice_z]
	# Look for stresses for configuration.  Assumes total stress is 14 lines down 
	# from where FORCE on cell is found 	
	if 'FORCE on cell' in line:
		line = lines[i+14]
		columns = line.split()
		stress_xx = float(columns[2])
		stress_yy = float(columns[3])
		stress_zz = float(columns[4])
		stress_xy = float(columns[5])
		stress_yz = float(columns[6])
		stress_zx = float(columns[7])
		stress_allx = [stress_xx, stress_xy, stress_zx]
		stress_ally = [stress_xy, stress_yy, stress_yz]
		stress_allz = [stress_zx, stress_yz, stress_zz]
		stress_component = [stress_allx, stress_ally, stress_allz]
	# Look for positions and forces for configuration
	if 'TOTAL-FORCE (eV/Angst)' in line:

		atom_cords = []
		atom_force = []
		line = lines[i+2]
		count = 0
		for count in range(1,natoms+1):
			columns = line.split()
			x_cord = float(columns[0])
			y_cord = float(columns[1])
			z_cord = float(columns[2])
			f_x = float(columns[3])
			f_y = float(columns[4])
			f_z = float(columns[5])

			coordinates = [x_cord, y_cord, z_cord]
			forces = [f_x, f_y, f_z]

			atom_cords.append(coordinates)
			atom_force.append(forces)
	
			line = lines[i+2+count]
	# Look for total energy of configuration.  Assumes that value is four
	# lines past where FREE ENERGIE OF THE ION-ELECTRON SYSTEM is found

	if 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
		data = {}		
		line = lines[i+4]
		columns = line.split()
		totalEnergy = float(columns[3])
		
	# Here is where all the data is put together since the energy value is the last
	# one listed in each configuration.  After this, all these values will be overwritten 
	# once the next configuration appears in the sequence when parsing
	
		data['Positions'] = atom_cords
		data['Forces'] = atom_force
		data['Stress'] = stress_component
		data['Lattice'] = all_lattice
		data['Energy'] = totalEnergy
		data['AtomTypes'] = listAtomTypes
		data['NumAtoms'] = natoms

	# Specify jsonfilename and put this and data into the write_json function.  All
	# json files should be output now.  The configuration number will be increased by one 
	# to keep track of which configuration is associated with which json file.
	
		jsonfilename = JSON_file + str(outcar_config_number) + '.json'
				
		write_json(data,jsonfilename)
		
		outcar_config_number = outcar_config_number + 1


	
		

		
