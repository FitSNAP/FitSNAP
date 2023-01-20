"""
This script parses through a single vasprun.xml file from VASP.
This file may include one or more configurations.
The output includes an XYZ file that can be used with FitSNAP.

Usage:
    python xml2xyz.py path/to/vasprun.xml path/to/file.xyz sample_every

where sample_every specifies to collect data every that many timesteps.

Alternatively, if you want to parse many vasprun.xml files in a single directory, where each 
vasprun.xml contains a single configuration, do

    python xml2xyz/py path/to/directory path/to/file.xyz sample_every

where sample_every is now a meaningless parameter.
Here, all data in the vasprun.xml files will be concatenated in a single file.xyz

TODO: Merge this script into the xml2JSON converter.
"""

import sys, os
from numpy import dot
import xml.etree.ElementTree as ET

xml_file = sys.argv[1]
xyz_filename = str(sys.argv[2])
sample_every = int(sys.argv[3]) # sample configurations every this many timesteps

fh = open(xyz_filename, 'w')

write_unconverged_steps_anyway = False

def write_xyz(atom_coords, atom_force, stress_component, all_lattice, totalEnergy, 
              listAtomTypes, natoms):
    """
    Write a single configuration to the XYZ file for this vasprun.xml file
    """

    #print(f"natoms: {natoms}")

    fh.write(f"{natoms}\n")
    line = f'Lattice = "{all_lattice[0][0]} {all_lattice[0][1]} {all_lattice[0][2]} \
                        {all_lattice[1][0]} {all_lattice[1][1]} {all_lattice[1][2]} \
                        {all_lattice[2][0]} {all_lattice[2][1]} {all_lattice[2][2]}"'
    line += " Properties=species:S:1:pos:R:3:forces:R:3 "
    line += f"energy={totalEnergy} "
    if (stress_component == []):
        line += f'stress = "0 0 0 0 0 0 0 0 0"\n'
    else:
        line += f'stress = "{stress_component[0][0]} {stress_component[0][1]} {stress_component[0][2]} \
                            {stress_component[1][0]} {stress_component[1][1]} {stress_component[1][2]} \
                            {stress_component[2][0]} {stress_component[2][1]} {stress_component[2][2]}"\n'

    fh.write(line)
    
    for n in range(0,natoms):
        fh.write(f"{listAtomTypes[n]} {atom_coords[n][0]} {atom_coords[n][1]} {atom_coords[n][2]} {atom_force[n][0]} {atom_force[n][1]} {atom_force[n][2]}\n")

order_atom_types = []
listAtomTypes = []
list_POTCARS = []
config_number = 1

# Start parsing through vasprun.xml looking for entries that are associated with the
# different values for the data needed, such as forces or positions

print(xml_file)

is_file = os.path.isfile(xml_file)
is_dir = os.path.isdir(xml_file)

print(is_file)
print(is_dir)

if (is_file):
    print("Supplied a file")
    tree = ET.iterparse(xml_file, events=['start', 'end'])
    for event, elem in tree:
        if elem.tag == 'parameters' and event=='end': #once at the start
            NELM = int(elem.find('separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]').text)
        elif elem.tag == 'atominfo' and event == 'end': #once at the start
            for entry in elem.find("array[@name='atoms']/set"):
                listAtomTypes.append(entry[0].text.strip())
            natoms = len(listAtomTypes)
            for entry in elem.find("array[@name='atomtypes']/set"):
                list_POTCARS.append(entry[4].text.strip().split())
            
        elif (elem.tag == 'structure' and not elem.attrib.get('name')) and event=='end': #only the empty name ones - not primitive cell, initial, or final (those are repeats) - so each ionic step
            all_lattice = []
            for entry in elem.find("crystal/varray[@name='basis']"):
                lattice_row = [float(x) for x in entry.text.split()]
                all_lattice.append(lattice_row)
            frac_atom_coords = []
            for entry in elem.find("varray[@name='positions']"):
                frac_atom_coords.append([float(x) for x in entry.text.split()])
            atom_coords = dot(frac_atom_coords, all_lattice).tolist()
            
        elif elem.tag == 'calculation' and event=='end': #this triggers each ionic step
            atom_force = []
            force_block = elem.find("varray[@name='forces']")
            if force_block:
                for entry in force_block:
                    atom_force.append([float(x) for x in entry.text.split()])
            stress_block = elem.find("varray[@name='stress']")
            stress_component = []
            if stress_block:
                for entry in stress_block:
                    stress_component.append([float(x) for x in entry.text.split()])
            totalEnergy = float(elem.find('energy/i[@name="e_0_energy"]').text)  ##NOTE! this value is incorrectly reported by VASP in version 5.4 (fixed in 6.1), see https://www.vasp.at/forum/viewtopic.php?t=17839
            # ASE vasprun.xml io reader has a more complex workaround to get the correct energy - we can update to include if needed
            if len(elem.findall("scstep")) == NELM:
                electronic_convergence = False ##This isn't the best way to check this, but not sure if info is directly available. Could try to calculate energy diff from scstep entries and compare to EDIFF
            else:
                electronic_convergence = True

            if (electronic_convergence and (config_number % sample_every == 0) ):
                write_xyz(atom_coords, atom_force, stress_component, all_lattice, totalEnergy, 
                          listAtomTypes, natoms)

            config_number += 1 

            if (config_number % sample_every == 0):
                print(config_number)

elif is_dir:
    print("Supplied a directory, finding all xml files.")

    # Find all XML files in the given directory

    files = []
    for (dirpath, dirnames, filenames) in os.walk(xml_file):
        if filenames != []:
            files.append(dirpath + '/' + filenames[0])

    for single_file in files:
        print(single_file)
        tree = ET.iterparse(single_file, events=['start', 'end'])
        try:
            for event, elem in tree:
                if elem.tag == 'parameters' and event=='end': #once at the start
                    NELM = int(elem.find('separator[@name="electronic"]/separator[@name="electronic convergence"]/i[@name="NELM"]').text)
                    
                elif elem.tag == 'atominfo' and event == 'end': #once at the start
                    listAtomTypes = []
                    for entry in elem.find("array[@name='atoms']/set"):
                        listAtomTypes.append(entry[0].text.strip())
                    natoms = len(listAtomTypes)
                    for entry in elem.find("array[@name='atomtypes']/set"):
                        list_POTCARS.append(entry[4].text.strip().split())
                    
                elif (elem.tag == 'structure' and not elem.attrib.get('name')) and event=='end': #only the empty name ones - not primitive cell, initial, or final (those are repeats) - so each ionic step
                    all_lattice = []
                    for entry in elem.find("crystal/varray[@name='basis']"):
                        lattice_row = [float(x) for x in entry.text.split()]
                        all_lattice.append(lattice_row)
                    frac_atom_coords = []
                    for entry in elem.find("varray[@name='positions']"):
                        frac_atom_coords.append([float(x) for x in entry.text.split()])
                    atom_coords = dot(frac_atom_coords, all_lattice).tolist()
                    
                elif elem.tag == 'calculation' and event=='end': #this triggers each ionic step
                    atom_force = []
                    force_block = elem.find("varray[@name='forces']")
                    if force_block:
                        for entry in force_block:
                            atom_force.append([float(x) for x in entry.text.split()])
                    stress_block = elem.find("varray[@name='stress']")
                    stress_component = []
                    if stress_block:
                        for entry in stress_block:
                            stress_component.append([float(x) for x in entry.text.split()])
                    totalEnergy = float(elem.find('energy/i[@name="e_0_energy"]').text)  ##NOTE! this value is incorrectly reported by VASP in version 5.4 (fixed in 6.1), see https://www.vasp.at/forum/viewtopic.php?t=17839
                    # ASE vasprun.xml io reader has a more complex workaround to get the correct energy - we can update to include if needed
                    if len(elem.findall("scstep")) == NELM:
                        electronic_convergence = False ##This isn't the best way to check this, but not sure if info is directly available. Could try to calculate energy diff from scstep entries and compare to EDIFF
                    else:
                        electronic_convergence = True

                    if (electronic_convergence and (config_number % sample_every == 0) ):
                        write_xyz(atom_coords, atom_force, stress_component, all_lattice, totalEnergy, 
                                  listAtomTypes, natoms)

                    config_number += 1 

                    if (config_number % sample_every == 0):
                        print(config_number)

        except:
            print(f"Faulty file at {single_file}")
            pass

fh.close()
