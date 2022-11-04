#! /usr/bin/env/ python

# This script will parse through a single vasprun.xml file from VASP, which may include one or more configurations, and will print out
# a JSON file(s) that can then be read into fitSNAP.  To run this script, you will need to specify a vasprun.xml file, and
# the name of the JSON file(s) that will be output in the command line  --->  python VASP2JSON.py myvasprun.xmlfile myJSONfile

import sys, os
import json
from numpy import dot
import xml.etree.ElementTree as ET

xml_file = sys.argv[1]
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
    
    :param data: dict, parsed output of POSCAR and vasprun.xml files
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
config_number = 1


# Start parsing through vasprun.xml looking for entries that are associated with the
# different values for the data needed, such as forces or positions


tree = ET.iterparse(xml_file, events=['start', 'end'])
for event, elem in tree:
    if elem.tag == 'parameters' and event=='end': #once at the start
        NELM = int(elem.find('separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]').text)
        #print(NELM)
        
    elif elem.tag == 'atominfo' and event == 'end': #once at the start
        for entry in elem.find("array[@name='atoms']/set"):
            listAtomTypes.append(entry[0].text.strip())
        natoms = len(listAtomTypes)
        #print('atom types', listAtomTypes)
        for entry in elem.find("array[@name='atomtypes']/set"):
            list_POTCARS.append(entry[4].text.strip().split())
        #print('potcars:', list_POTCARS)
        
    elif (elem.tag == 'structure' and not elem.attrib.get('name')) and event=='end': #only the empty name ones - not primitive cell, initial, or final (those are repeats) - so each ionic step
        all_lattice = []
        for entry in elem.find("crystal/varray[@name='basis']"):
            lattice_row = [float(x) for x in entry.text.split()]
            all_lattice.append(lattice_row)
        #print('lattice = ', all_lattice)
        frac_atom_coords = []
        for entry in elem.find("varray[@name='positions']"):
            frac_atom_coords.append([float(x) for x in entry.text.split()])
        atom_coords = dot(frac_atom_coords, all_lattice).tolist()
        #print(atom_coords)
        
    elif elem.tag == 'calculation' and event=='end': #this triggers each ionic step
        atom_force = []
        force_block = elem.find("varray[@name='forces']")
        if force_block:
            for entry in force_block:
                atom_force.append([float(x) for x in entry.text.split()])
        #print(atom_force)
        stress_block = elem.find("varray[@name='stress']")
        stress_component = []
        if stress_block:
            for entry in stress_block:
                stress_component.append([float(x) for x in entry.text.split()])
        totalEnergy = float(elem.find('energy/i[@name="e_0_energy"]').text)  ##NOTE! this value is incorrectly reported by VASP in version 5.4 (fixed in 6.1), see https://www.vasp.at/forum/viewtopic.php?t=17839
        ## ASE vasprun.xml io reader has a more complex workaround to get the correct energy - we can update to include if needed
        #print(totalEnergy)
        if len(elem.findall("scstep")) == NELM:
            electronic_convergence = False ##This isn't the best way to check this, but not sure if info is directly available. Could try to calculate energy diff from scstep entries and compare to EDIFF
        else:
            electronic_convergence = True
        
        # Here is where all the data is put together for each ionic step
        # After this, all these values will be overwritten
        # once the next configuration appears in the sequence when parsing
        data = {}
        data["Positions"] = atom_coords
        if atom_force:
            data["Forces"] = atom_force
        if stress_component:
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

        jsonfilename = JSON_file + str(config_number) + ".json"

        if electronic_convergence:
            write_json(data, jsonfilename)
        else:
            if write_unconverged_steps_anyway:
                write_json(data, jsonfilename)

        config_number += 1 
