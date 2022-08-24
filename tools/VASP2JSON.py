#! /usr/bin/env/ python

# This script will parse through a single OUTCAR file from VASP, which may include one or more configurations, and will print out
# a JSON file(s) that can then be read into fitSNAP.  To run this script, you will need to specify an OUTCAR file, and
# the name of the JSON file(s) that will be output in the command line  --->  python VASP2JSON.py myOUTCARfile myJSONfile

import sys, os
import json

OUTCAR_file = sys.argv[1]
JSON_file = str(sys.argv[2])

write_unconverged_steps_anyway = False

def write_json(data, jsonfilename):
    """
    Define a function to take all the data that will be needed and send that to a
    json.dump command which then takes that data and puts it into json format. 
    The input for this function (data) is a dictionary that holds all the values from
    the OUTCAR file that will be parsed through (such as atom positions, forces, lattice
    vectors, etc.) which will be put into all Data with key 'Data'.  allDataHeader holds
    all of the units as well as 'Data'.  This is then encompassed by myDataSet which then gets
    fed into the json.dump command.  Most of this is just getting everything into the right format
    
    :param data: dict, parsed output of POSCAR and OUTCAR files
    :param jsonfilename: str, filename of .json file to be written 
    """
    jsonfile = open(jsonfilename, "w")
    # print >>jsonfile, "# Header\n"
    # print("# Header", file=jsonfilename)
    allData = [data]
    allDataHeader = {}
    allDataHeader["EnergyStyle"] = "electronvolt"
    allDataHeader["StressStyle"] = "kB"
    allDataHeader["AtomTypeStyle"] = "chemicalsymbol"
    allDataHeader["PositionsStyle"] = "angstrom"
    allDataHeader["ForcesStyle"] = "electronvoltperangstrom"
    allDataHeader["LatticeStyle"] = "angstrom"
    allDataHeader["Data"] = allData

    myDataset = {}

    myDataset["Dataset"] = allDataHeader
    #jsonfile.write(json.dumps(myDataset))  # if you want a condensed string
    json.dump(myDataset, jsonfile, indent=2, sort_keys=True)  #if you want the expanded, multi-line format
    jsonfile.close()
    return


order_atom_types = []
listAtomTypes = []
list_POTCARS = []

# Start parsing through OUTCAR looking for keywords that are assocaited with the
# different values for the data needed, such as forces or positions

with open(OUTCAR_file, "rt") as f2:
    lines = f2.readlines()
outcar_config_number = 1

for i, line in enumerate(lines):
    # Look for the ordering of the atom types - grabbing POTCAR filenames first, then atom labels separately because VASP has terribly inconsistent formatting
    if "POTCAR:" in line:
        if (
            line.split()[1:] not in list_POTCARS
        ):  # VASP will have these lines in the OUTCAR twice, and we don't need to append them the second time
            list_POTCARS.append(
                line.split()[1:]
            )  # each line will look something like ['PAW_PBE', 'Zr_sv_GW', '05Dec2013']
    # can grab atom labels from VRHFIN lines at top of OUTCAR - much more consistent formatting than the POTCAR names
    # (These will only show up once for each element type in the OUTCAR)
    elif "VRHFIN" in line:
        order_atom_types.append(line.split()[1][1:].strip(" :"))
    # Look for number of atoms in configuration
    elif "number of ions" in line:
        columns = line.split()
        natoms = int(columns[11])
    # Look for the number of atoms of each element type in configuration
    elif "ions per type =" in line:
        num_atoms_per_type = [int(num) for num in line.split()[4:]]
        num_atom_types = len(num_atoms_per_type)
        assert (
            len(order_atom_types) == num_atom_types
        ), "number of element types and length of element list disagree"
        # make the atom_types list for the json file
        for atom_type in range(0, num_atom_types):
            totalAtoms = int(num_atoms_per_type[atom_type])
            atomType = str(order_atom_types[atom_type])
            for n in range(0, totalAtoms):
                listAtomTypes.append(atomType)

    elif "aborting loop because EDIFF is reached" in line:
        electronic_convergence = True
    elif "aborting loop EDIFF was not reached (unconverged)" in line:
        electronic_convergence = False
                
    # Look for lattice vectors for configuration
    elif "direct lattice vectors" in line:
        lattice_x = [float(x) for x in lines[i + 1].split()[0:3]]
        lattice_y = [float(y) for y in lines[i + 2].split()[0:3]]
        lattice_z = [float(z) for z in lines[i + 3].split()[0:3]]
        all_lattice = [lattice_x, lattice_y, lattice_z]

    # Look for stresses for configuration.  Assumes total stress is 14 lines down
    # from where FORCE on cell is found
    elif "FORCE on cell" in line:
        stress_xx, stress_yy, stress_zz, stress_xy, stress_yz, stress_zx = [
            float(s) for s in lines[i + 14].split()[2:8]
        ]
        stress_allx = [stress_xx, stress_xy, stress_zx]
        stress_ally = [stress_xy, stress_yy, stress_yz]
        stress_allz = [stress_zx, stress_yz, stress_zz]
        stress_component = [stress_allx, stress_ally, stress_allz]
    # Look for positions and forces for configuration
    elif "TOTAL-FORCE (eV/Angst)" in line:

        atom_cords = []
        atom_force = []
        for count in range(0, natoms):
            x_cord, y_cord, z_cord, f_x, f_y, f_z = [
                float(val) for val in lines[i + 2 + count].split()[0:6]
            ]

            coordinates = [x_cord, y_cord, z_cord]
            forces = [f_x, f_y, f_z]

            atom_cords.append(coordinates)
            atom_force.append(forces)

    # Look for total energy of configuration.  Assumes that value is four
    # lines past where FREE ENERGIE OF THE ION-ELECTRON SYSTEM is found

    elif "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
        data = {}
        totalEnergy = float(lines[i + 4].split()[3])

        # Here is where all the data is put together since the energy value is the last
        # one listed in each configuration.  After this, all these values will be overwritten
        # once the next configuration appears in the sequence when parsing

        data["Positions"] = atom_cords
        data["Forces"] = atom_force
        data["Stress"] = stress_component
        data["Lattice"] = all_lattice
        data["Energy"] = totalEnergy
        data["AtomTypes"] = listAtomTypes
        data["NumAtoms"] = natoms
        data["computation_code"] = "VASP"
        data["pseudopotential_information"] = list_POTCARS

        # Specify jsonfilename and put this and data into the write_json function.  All
        # json files should be output now.  The configuration number will be increased by one
        # to keep track of which configuration is associated with which json file.

        jsonfilename = JSON_file + str(outcar_config_number) + ".json"

        if electronic_convergence:
            write_json(data, jsonfilename)
        else:
            if write_unconverged_steps_anyway:
                write_json(data, jsonfilename)

        outcar_config_number = outcar_config_number + 1
