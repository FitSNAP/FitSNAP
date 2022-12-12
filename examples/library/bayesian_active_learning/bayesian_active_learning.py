"""
Bayesian active learning script using the FitSNAP library.

Usage:
    python -m bayesian_active_learning.py --fitsnap_in Ta-example.in

Variables in this script to change:
    plot_stuff : determines whether uncertainties are plotted
"""

import numpy as np
import configparser
import argparse
from datetime import datetime
from os import chdir, mkdir, getenv, getcwd, path
import json
from subprocess import run
from shutil import copyfileobj
import inspect
import copy 
import pandas as pd
import logging

plot_stuff = True
if plot_stuff:
    import matplotlib.pyplot as plt
known_truth_for_unlabeled = True

class AL_settings_class():
    """
    This loads all the settings needed from the config_parser object, filling them with default values if they don't exist. 
    It doesn't check for any extraneous or nonsensical values.
    """
    def __init__(self, AL_config):
        self.number_of_iterations = AL_config.getint('GENERAL', 'number_of_iterations', fallback = 10)
        self.cluster_structures = AL_config.getboolean('GENERAL', 'cluster_structures', fallback = False)
        self.batch_size = AL_config.getint('GENERAL', 'batch_size', fallback = 1)
        self.training_path = AL_config.get('GENERAL', 'training_path', fallback = None)
        self.unlabeled_path = AL_config.get('GENERAL', 'unlabeled_path', fallback = None)
        self.E_weight = AL_config.getfloat('OBJECTIVE', 'E_weight', fallback = 1.0)
        self.F_weight = AL_config.getfloat('OBJECTIVE', 'F_weight', fallback = 1.0)
        self.S_weight = AL_config.getfloat('OBJECTIVE', 's_weight', fallback = 1.0)
        self.EFS_reweighting = [self.E_weight, self.F_weight, self.S_weight]
        self.F_agg_function = AL_config.get('OBJECTIVE', 'F_aggregation_function', fallback = None)
        self.S_agg_function = AL_config.get('OBJECTIVE', 'S_aggregation_function', fallback = None)
        self.FS_agg_functions = [self.F_agg_function, self.S_agg_function]
        self.obj_function = AL_config.get('OBJECTIVE', 'objective_function', fallback = 'sum')
        self.weight_by_relative_DFT_cost = AL_config.getboolean('OBJECTIVE', 'weight_by_relative_DFT_cost', fallback = True)
        
def deepcopy_pt_internals(snap):
    """
    Deep copy SNAP arrays from the parallel tools shared arrays.
    """
    pt = snap.pt
    i1, i2 = pt.fitsnap_dict['sub_a_indices']
    i2 += 1  
    a_len = pt.shared_arrays['a'].array.shape[0]

    # obtain separated dictionary names else silent runtime error when dict changes size

    shared_array_names = list(pt.shared_arrays.keys())
    for array_name in shared_array_names:
        array_dims = pt.shared_arrays[array_name].array.shape
        if array_dims[0] == a_len:
            if len(array_dims) == 2:
                pt.create_shared_array(array_name+'_copy', array_dims[0], array_dims[1], tm=snap.config.sections["SOLVER"].true_multinode)
            elif len(array_dims) == 1:
                pt.create_shared_array(array_name+'_copy', array_dims[0], tm=snap.config.sections["SOLVER"].true_multinode)
            else:
                raise Error('I did not code for more than 2d arrays.')
            if rank == 0:
                print(array_name+'_copy has been instantiated')

            # fill the cloned arrays - each processor should be doing the segment it originally handled

            if parallel:
                comm.Barrier()
            pt.shared_arrays[array_name+'_copy'].array[i1:i2] = copy.deepcopy(pt.shared_arrays[array_name].array[i1:i2])
            if parallel:
                comm.Barrier()
            if rank == 0:
                assert np.allclose(pt.shared_arrays[array_name+'_copy'].array, pt.shared_arrays[array_name].array)
                print(array_name+'_copy has been fully copied')
            del pt.shared_arrays[array_name].array
            del pt.shared_arrays[array_name]
            if parallel:
                comm.Barrier()
            if rank == 0:
                print('original ', array_name, ' has been deleted')
        else: 
            
            # just copy over the little lists on the head proc
            
            if len(array_dims) == 2:
                pt.create_shared_array(array_name+'_copy', array_dims[0], array_dims[1], dtype='i')
            elif len(array_dims) == 1:
                pt.create_shared_array(array_name+'_copy', array_dims[0], dtype='i')
            else:
                raise Error('I did not code for more than 2d arrays.')
            if rank == 0:
                print(array_name+'_copy has been instantiated')
                pt.shared_arrays[array_name+'_copy'].array = copy.deepcopy(pt.shared_arrays[array_name].array)
                assert np.allclose(pt.shared_arrays[array_name+'_copy'].array, pt.shared_arrays[array_name].array)
                print(array_name+'_copy has been fully copied')
            del pt.shared_arrays[array_name].array
            del pt.shared_arrays[array_name]
            if rank == 0:
                print('original ', array_name, ' has been deleted')
            #print('rank', rank,'reports pt arrays existing as:', pt.shared_arrays)
            if parallel:
                comm.Barrier()
        
    # copy our metadata - all of this is only on the head processor as process_configs() sent 
    # everything back to it with a calculator.collect_distrubuted_lists() call
    
    if rank == 0:
        fictsnap_dict_metadata_lists = [name for name in pt.fitsnap_dict.keys() if type(pt.fitsnap_dict[name]) is list]
        for list_name in fictsnap_dict_metadata_lists:
            print(list_name)
            print('rank', rank, 'reports pt.fitsnap_dict[', list_name, '] as length:',len(pt.fitsnap_dict[list_name]))
            #pt.add_2_fitsnap(list_name+"_copy", DistributedList(len(pt.fitsnap_dict[list_name])))
            pt.add_2_fitsnap(list_name+"_copy", copy.deepcopy(pt.fitsnap_dict[list_name]))
            print(list_name+"_copy has been instantiated")
        #from fitsnap3lib.parallel_tools import stubs  ## why is this not just a pt property?
        #if pt.fitsnap_dict[list_name+"_copy"] is not None and stubs != 1:
        #    pt.fitsnap_dict[list_name+"_copy"] = [item for sublist in pt.fitsnap_dict[list_name+"_copy"] for item in sublist]
        #elif pt.fitsnap_dict[list_name+"_copy"] is not None:
        #    pt.fitsnap_dict[list_name+"_copy"] = pt.fitsnap_dict[list_name+"_copy"].get_list()
        
            assert pt.fitsnap_dict[list_name+"_copy"] == pt.fitsnap_dict[list_name]
            print(list_name+"_copy has been fully copied")
            del pt.fitsnap_dict[list_name]
            print('original ', list_name, ' has been deleted')
    if parallel:
        comm.Barrier()


def objective_function(df, EFS_reweighting=[1.0, 1.0, 1.0], FS_agg_functions=[None, None], objective_function='sum', norm_force_components=False, weight_by_relative_DFT_cost = True):
    """
    Evaluates an objective function on all structures within the pandas dataframe
    Returns ranked structures

    Args:
        df: pandas dataframe, output of FitSNAP with 'uncertainty' column added
        nadd: int, the number of structures to choose
        EFS_reweighting : weights to apply to the uncertainty of E, F, and S rows - possibly also 
            allow % or other modes than straight multiplicative weights
        FS_agg_functions : functions to apply to F, S rows before inputting into objective function 
            - format is string name of any numpy function, or None - e.g. 'mean'
        objective_function : function to apply to each structure's collection of (potentially 
            modified by above variables) rows; options are 'sum, 'max', and 'average'
        norm_force_components : whether to turn the x,y,z components of the forces into a force 
            magnitude (and average the A matrix values)
            TODO: better options here, might be better to do sum than 2-norm
        weight_by_relative_DFT_cost: scale each structures objective function based off of 
            1/(number of atoms)^3 - rough approximation of DFT cost scaling
    
    Returns:
        chosen_structures: a pandas dataframe containing the group and config as multi-index of 
        the top nadd structures 
    """
    m_df = df.copy()  # the dataframe to be manipulated below, copying to be safe

    """
    First, get count of the number of atoms in each structure from number of forces.
    Can only be done this way if fitting forces, could rewrite to pull data from elsewhere in 
    snap object for energy only fitting.
    """

    if weight_by_relative_DFT_cost:
        m_df = m_df.join((m_df[m_df["Row_Type"]=="Force"].groupby(['Groups', 'Configs'], observed=True, sort=False).size()/3).astype(int).rename('num_atoms'), on=['Groups', 'Configs'])
        m_df['relative_cost_factor'] = m_df['num_atoms']**3
        m_df['uncertainty'] = m_df['uncertainty']/m_df['relative_cost_factor']
        
    """
    Second, convert x,y,z components of forces on each atom to force magnitudes (if desired).
    Note, might better to do sum rather than 2-norm.
    """

    if norm_force_components:
        d3 = {i : 'mean' for i in A_matrix_columns}   ## averaging the A matrix elements of the three force component rows TODO: give more options for how to handle this
        d2 = {i : 'first' for i in (set(df.columns) - set(A_matrix_columns) - {'Groups', 'Configs', 'Atom_I', 'uncertainty'})} ##grabbing the remaining columns - should all be the same for each atom - Rowtype, weights, test/train identifier
        FM_df = m_df[m_df["Row_Type"] == "Force"].groupby(['Groups', 'Configs', 'Atom_I'], observed=True).agg({'uncertainty':np.linalg.norm}|d3|d2)
        m_df = pd.concat(m_df[m_df["Row_Type"].isin(["Energy", "Stress"])], FM_df)
        del FM_df
        
    """
    Third, apply any desired aggregating functions to the force magnitudes/components and stress 
    components.
    """

    if FS_agg_functions[0] or FS_agg_functions[1]:
        d2 = {i : 'first' for i in set(df.columns) - set(A_matrix_columns) - {'Groups', 'Configs', 'Atom_I', 'uncertainty'}}
        if FS_agg_functions[0]:
            d1 = {'uncertainty' : getattr(np, FS_agg_functions[0])}
            # A matrix terms do not directly correlate to structure aggregated uncertainty predictions
            F_df = m_df[m_df["Row_Type"] == "Force"].groupby(['Groups', 'Configs'], observed=True).agg(d1|d2) 
        else:
            F_df = m_df[m_df["Row_Type"] == "Force"]
            
        if FS_agg_functions[1]:
            d1 = {'uncertainty' : getattr(np, FS_agg_functions[1])}
            # A matrix terms do not directly correlate to structure aggregated uncertainty predictions
            S_df = m_df[m_df["Row_Type"] == "Stress"].groupby(['Groups', 'Configs'], observed=True).agg(d1|d2)
        else:
            S_df = m_df[m_df["Row_Type"] == "Stress"]        
        m_df = pd.concat([m_df[m_df["Row_Type"]=="Energy"], F_df, S_df])
        del F_df
        del S_df
        
    """
    Fourth, apply an weighting effects to the EFS uncertainty predictions.
    """

    conditions = [m_df['Row_Type']=="Energy", m_df['Row_Type']=="Force", m_df['Row_Type']=="Stress"]
    values = EFS_reweighting
    m_df['uncertainty'] *= np.select(conditions, values)

    """
    Last, apply the objective function on the transformed df
    """

    if objective_function=='max': #simplest case - return the highest uncertainty row in each structure, taking the top nadd structures
        ranked_structures = m_df.sort_values("uncertainty", ascending=False, key=abs).groupby(['Groups', 'Configs'], observed=True, sort=False).first()
        #x_vector_for_each_structure = ranked_structures[A_matrix_columns].values  ## TODO: will need to update how I'm grabbing this when aggregating F or S rows
    elif objective_function=='average':
        ranked_structures = m_df.groupby(['Groups', 'Configs'], observed=True, sort=False).agg({'uncertainty':np.average}).sort_values("uncertainty", ascending=False, key=abs)
        #x_vector_for_each_structure = np.concatenate([df[(df['Row_Type']=='Energy') & (df['Groups']==group) &\
        #                                                 (df['Configs']==config)][A_matrix_columns].values for group, config in ranked_structures.index]) #just using the energy A-matrix line for this case, note - pulling from the original df, not m_df!
    elif objective_function=='sum':
        ranked_structures = m_df.groupby(['Groups', 'Configs'], observed=True, sort=False).agg({'uncertainty':np.sum}).sort_values("uncertainty", ascending=False, key=abs)
        #x_vector_for_each_structure = np.concatenate([df[(df['Row_Type']=='Energy') & (df['Groups']==group) &\
        #                                                 (df['Configs']==config)][A_matrix_columns].values for group, config in ranked_structures.index]) #just using the energy A-matrix line for this case, note - pulling from the original df, not m_df!
    else:
        print("Specified objective function does not match any coded options!")

    return ranked_structures#, x_vector_for_each_structure


class VASP_runner():
    def __init__(self, AL_config, AL_settings):
        self.config = AL_config
        self.settings = AL_settings
        timestamp = datetime.now()
        self.VASP_working_directory = timestamp.strftime('%Y-%m-%d__%H-%M-%S')+'__run_VASP_calculations'
        self.config['VASP']['VASP_working_directory'] = self.VASP_working_directory
        self.settings.VASP_working_directory = self.VASP_working_directory
        mkdir(self.VASP_working_directory)
        self.job_count = 0
        self.nodes = int(getenv('nodes', 1))
        print('The VASP_runner class found it has', self.nodes, 'nodes to use. If incorrect, make sure to export the "nodes" environment variable.')
        self.cores_per_node = int(getenv('cores', 1))
        print('The VASP_runner class found it has', self.cores_per_node, 'cores to use per node. If incorrect, make sure to export the "cores" environment variable.')
        
        self.settings.VASP_executable_path = self.config.get('VASP', 'VASP_executable_path',
                                                      fallback='/projects/vasp/2020-build/clusters/vasp6.1.1/bin/vasp_std') # location on Attaway and similar; will need to be set on other clusters
        self.settings.VASP_kpoints_auto_generation_Rk = self.config.getint('VASP', 'VASP_kpoints_auto_generation_Rk',
                                                                 fallback = 30)  #can go as low as 10 for large gap insulators or as high as 100 for metals; see https://www.vasp.at/wiki/index.php/KPOINTS
        ## TODO: add option to read from training data and match if value specified in the training set

        #if "VASP_pseudopotential_library" not in input_dict:
        #    sys.exit("You must supply the path to the pseudopotential library as an input labeled 'VASP_pseudopotential_library'.")
        #TODO: add in some ability to read .jsons from the existing training data use their listed POTCAR names if none given in input

    def __call__(self, filepaths, input_style='json'):
        output_directories = []
        top_dir = getcwd()
        for filepath in filepaths:
            ## make and move into the calculation directory
            VASP_job_directory = str(self.job_count)
            self.job_count += 1
            mkdir(self.VASP_working_directory + '/' + VASP_job_directory)
            chdir(self.VASP_working_directory + '/' + VASP_job_directory)
            
            ## load in json information
            with open(top_dir + '/' + filepath) as file:
                file.readline()
                try:
                    data = loads(file.read(), parse_constant=True)
                except Exception as e:
                    print("Trouble Parsing Json Data: ", filepath)
                assert len(data) == 1, "More than one object (dataset) is in this file"
                data = data['Dataset']
                assert len(data['Data']) == 1, "More than one configuration in this dataset"
                data['Group'] = filepath.split("/")[-2]
                data['File'] = filepath.split("/")[-1]
                assert all(k not in data for k in data["Data"][0].keys()), "Duplicate keys in dataset and data"
                data.update(data.pop('Data')[0])  # Move data up one level
            ## make POSCAR - this (and the above section) is basically the json_to_POSCAR.py script
            with open('POSCAR', 'w') as poscar:
                poscar.write(data['Group']+'/'+data['File']+" converted to POSCAR \n")
                poscar.write('1.0 \n')
                lattice_vectors = data["Lattice"]
                poscar.write(str(lattice_vectors[0][0]) + ' ' + str(lattice_vectors[0][1]) + ' ' + str(lattice_vectors[0][2]) + '\n')
                poscar.write(str(lattice_vectors[1][0]) + ' ' + str(lattice_vectors[1][1]) + ' ' + str(lattice_vectors[1][2]) + '\n')
                poscar.write(str(lattice_vectors[2][0]) + ' ' + str(lattice_vectors[2][1]) + ' ' + str(lattice_vectors[2][2]) + '\n')
                atom_types_list = data["AtomTypes"]
                atom_types_set = set(atom_types_list)
                atom_type_ordered_list = list(atom_types_set)
                atom_type_ordered_list.sort()
                n_atoms_per_type = [0]*len(atom_type_ordered_list)
                for i in range(0,len(atom_types_list)):
                    for j in range(0, len(atom_type_ordered_list)):
                        if atom_types_list[i] == atom_type_ordered_list[j]:
                            n_atoms_per_type[j] += 1

                POSCAR_atom_types_line = ''
                POSCAR_n_atoms_per_type_line = ''
                for i in range(len(atom_type_ordered_list)):
                    POSCAR_atom_types_line += ' ' + atom_type_ordered_list[i]
                    POSCAR_n_atoms_per_type_line += ' ' + str(n_atoms_per_type[i])
                poscar.write(POSCAR_atom_types_line + '\n')
                poscar.write(POSCAR_n_atoms_per_type_line + '\n')
                poscar.write("Cartesian \n")
                current_positions = data["Positions"]
                for j in range(0, len(atom_type_ordered_list)): # loop once for each type of atom
                    for i in range(0,len(current_positions)):  #check each atom in entire list
                        if atom_types_list[i] == atom_type_ordered_list[j]: #if atom is the type of atom currently looking for
                            poscar.write(str(current_positions[i][0]) + ' ' + str(current_positions[i][1]) + ' ' + str(current_positions[i][2]) + '\n') #then add atom to POSCAR file

            # make POTCAR - currently relying on user input paths
            # TODO: add in option for defaults for elements and just a path to the VASP pseudopotential library
            # TODO: add in ability to read training JSONs and parse to see if they have POTCAR data

            element_POTCAR_paths = []
            for element in atom_type_ordered_list:
                try:
                    element_POTCAR_paths.append(self.config['VASP'][element+'_POTCAR_location']) 
                except: #failure to find POTCAR location in the inputs for the given atom type
                    print("ERROR! No POTCAR filepath give for atoms of type", element)
                    print("You must include a filepath in the Active Learning input file under the VASP heading named", str(element)+"_POTCAR_location")

            with open('POTCAR', 'wb') as potcar:
                for elem_POTCAR_path in element_POTCAR_paths:
                    with open(elem_POTCAR_path, 'rb') as e_POT_file:
                        copyfileobj(e_POT_file, potcar)

            # copy or create KPOINTS file

            if exists('../../KPOINTS'):
                print('Using the existing KPOINTS file in the top active learning directory')
                with open('KPOINTS', 'wb') as KPOINTS:
                    with open('../../KPOINTS', 'rb') as source_KPOINTS:
                        copyfileobj(source_KPOINTS, KPOINTS)
            else:
                print('Creating a KPOINTS file with an Rk of', self.settings.VASP_kpoints_auto_generation_Rk)
                with open('KPOINTS', 'w') as KPOINTS:
                    KPOINTS.write("K-Points\n 0\n Auto\n ")
                    KPOINTS.write(str(self.settings['VASP']["VASP_kpoints_auto_generation_Rk"])+"\n")
                              
            # copy or make INCAR, but adjust settings that need to change with system

            if exists('../../INCAR'):
                print('Using the existing INCAR file in the top active learning directory')
                with open('INCAR', 'w') as INCAR:
                    with open('../../INCAR', 'r') as source_INCAR:
                        lines = source_INCAR.readlines()
                        for i, line in enumerate(lines):
                            if "MAGMOM" in line:
                                # Just give everything an initial magnetic moment of 4 and let things relax down
                                lines[i] = "MAGMOM = " + str(len(current_positions)) + "*4 \n"
                    INCAR.write(''.join(lines))
            else:
                print('Creating an INCAR file from scratch, ENCUT determined by POTCAR file')
                with open('INCAR', 'w') as INCAR:
                    INCAR.write("IBRION = 2 \n")  # Relaxation calculation
                    INCAR.write("ISIF = 3 \n")    # Ionic and cell shape and cell volume all relax
                    INCAR.write("EDIFF = 1.0e-06 \n")  # Generally high enough accuracy for anything not going beyond basic DFT (e.g. phonons, GW, BSE, etc.)
                    INCAR.write("EDIFFG = -0.02 \n")   # Stops ionic relaxation when forces on all atoms below abs(value)
                    INCAR.write("ISPIN = 2 \n")      # Perform spin polarized calculations
                    INCAR.write("MAGMOM = " + str(len(current_positions)) + "*4 \n")  # Just give everything an initial magnetic moment of 4 and let things relax down
                    INCAR.write("PREC = Accurate \n")  # Good to be safe.
                    # TODO: The ENCUT should be determined once and saved for all future passes through the oracle.
                    # TODO: After setting up ability to use default or file-read POTCARs, will need to adjust this section
                    largest_ENMAX = 0
                    for dict_key, dict_val in self.config['VASP'].items():
                        if dict_key.endswith("_POTCAR_location"):
                            with open(dict_val) as e_POT_file:
                                for line in e_POT_file:
                                    if "ENMAX" in line:
                                        ENMAX = float(line.split()[2].strip(";"))
                                        if ENMAX > largest_ENMAX:
                                            largest_ENMAX = ENMAX
                                        break  #no reason to keep reading file after locating ENMAX line
                    INCAR.write("ENCUT = "+str(largest_ENMAX)) #taking the highest ENMAX value from all POTCARs provided in the input.txt file
                    INCAR.write("ALGO = Fast \n")     # Usually stable, can change to normal if having problems.
                    INCAR.write("ISMEAR = 0 \n")     # Most generally applicable smearing method, whether for metals or insulators.
                    INCAR.write("SIGMA = 0.03 \n")   # This is a small sigma. It should be good for insulators. It may not be the best for metals.
                    INCAR.write("LWAVE = .FALSE. \n")  # Do not write out the wavefunctions to the WAVECAR file.
                    INCAR.write("LCHARG = .FALSE. \n") # Do not write out the charge density to the CHGCAR and CHG files.
                    # This seems generally a good set of parallelization settings, and avoids some instability issues in VASP 6 on the Attaway cluster
                    # TODO: maybe do some more intelligent parallelization settings based on number of atoms and size of unit cell?
                    INCAR.write("NCORE = "+str(self.cores_per_node)) 
                    INCAR.write("KPAR = "+str(self.nodes))
                    
            # run VASP in current instance

            args = ['mpiexec', '--bind-to', 'core', '--npernode', str(self.cores_per_node), '--n', str(self.nodes*self.cores_per_node), self.settings.VASP_executable_path]
            print("running command:", args)
            run(args)
                
            # TODO: some error checking to confirm the VASP job ran successfully

            # return to top directory and add directory to list to be returned for the running of VASP2JSON.py on

            chdir('../..')
            output_directories.append(self.VASP_working_directory + '/' + VASP_job_directory)  #may want to do absolute paths instead of relative paths
            
        return output_directories

# import parallel tools and create pt object
# this is the backbone of FitSNAP
# use mpi if available
try:
    import mpi4py as mpi4py
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
    pt = ParallelTools()
    parallel = True
except ModuleNotFoundError:
    from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
    pt = ParallelTools()
    rank = 0
    parallel = False

"""
Read user supplied input scripts and create config objects.
We have two config objects here - `config` is the typical FitSNAP object while `AL_config` will 
read from a custom active learning config file, to be impelemented later. This AL config file will
contain settings such as number of iterations used in active learning.
"""

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default='fitsnap.in')
parser.add_argument("--AL_in", help="Active learning input script.", default='AL.in')
args = parser.parse_args()
if rank == 0:
    last_timestamp = datetime.now()
    print("FitSNAP input script:")
    print(args.fitsnap_in)
    print("Active Learning input script:")
    print(args.AL_in)
AL_config = configparser.ConfigParser()
AL_config.read(args.AL_in)
AL_settings = AL_settings_class(AL_config)

if rank==0:
    print("----- main script")
    current_timestamp = datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
# Config class reads the input from the parsed fitsnap.in file
from fitsnap3lib.io.input import Config
if parallel:
    comm.Barrier()
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
directory = config.sections['PATH'].datapath.split('/')[0:-1]
if AL_settings.unlabeled_path:
    config.sections['PATH'].datapath = AL_settings.unlabeled_path
else:
    AL_settings.unlabeled_path = '/'.join(directory + ['unlabeled_JSON/'])
    config.sections['PATH'].datapath = AL_settings.unlabeled_path
    
for key in list(config.sections['GROUPS'].group_table.keys()):
    if not path.isdir(config.sections['PATH'].datapath+'/'+key):
        config.sections['GROUPS'].group_table.pop(key)

# create a fitsnap object - uses the previously defined pt and config objects

from fitsnap3lib.fitsnap import FitSnap
if parallel:
    comm.Barrier()
snap = FitSnap()
if rank==0:
    print('first snap object made')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp

# tell ParallelTools not to check for existing fitsnap objects

pt.check_fitsnap_exist = False

snap.scraper.scrape_groups()
if parallel:
    comm.Barrier()
if rank==0:
    print('1st scrape_groups() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.scraper.divvy_up_configs()
if parallel:
    comm.Barrier()
if rank==0:
    print('1st divvy_up_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.data = snap.scraper.scrape_configs()
if parallel:
    comm.Barrier()
if rank==0:
    print('1st scrape_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.calculator.shared_index=0
snap.calculator.distributed_index=0
if parallel:
    comm.Barrier()
snap.process_configs()

"""
These are only being done to access the pandas dataframe.
Should update the fitsnap calculate section that can dump a dataframe to also have a function for 
returning a handle to a pandas dataframe, then can avoid doing the fit step.
"""

if parallel:
    comm.Barrier()
if rank==0:
    print('1st process_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.solver.perform_fit()
if parallel:
    comm.Barrier()
if rank==0:
    print('1st perform_fit() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.solver.fit_gather()
if parallel:
    comm.Barrier()
if rank==0:
    print('1st fit_gather() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.solver.error_analysis()
if rank==0:
    print('1st error_analysis() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
if parallel:
    comm.Barrier()
deepcopy_pt_internals(snap) 

if parallel:
    comm.Barrier()
if rank == 0:
    print('pt internals deepcopy done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
    metadata_labels_for_objective_function = ['Groups_copy', 'Configs_copy', 'Atom_I_copy', 'Row_Type_copy', 'Atom_Type_copy'] #Atom_Type_copy not actually needed here, could be left out 
    unlabeled_df = pd.DataFrame()
    for key in metadata_labels_for_objective_function:
        # TODO: Take these out of the pt internals deepcopying and just put them into the dataframe
        #       immediately, if sticking to the df for metadata
        unlabeled_df[key[0:-5]] = pd.Categorical(pt.fitsnap_dict[key])
    mask_of_still_unused = [True]*len(pt.shared_arrays['a_copy'].array)

if parallel:
    comm.Barrier()

# need to del and reinstantiate pt and config here - modified during run to add training set sizes 
# and other changes
# TODO: could be nice to make a .reset functionality in them to return them to state just after 
#       instantiation (after taking input but before process configs)

del snap
del config
del pt
if parallel:
    comm.Barrier()
pt = ParallelTools()
if parallel:
    comm.Barrier()
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])

if AL_settings.training_path:
    config.sections['PATH'].datapath = AL_settings.training_path
else:
    config.sections['PATH'].datapath = '/'.join(directory + ['training_JSON/'])
# switch out our solver to the ANL solver to get the covariance matrix that we need.
config.sections['SOLVER'].solver = 'ANL'

if path.isdir(config.sections['PATH'].datapath+'/testing_json_group/'):
    config.sections['GROUPS'].group_table['testing_json_group'] = {'training_size': 0.0, 'testing_size': 1.0, 'eweight': 1.0, 'fweight': 1.0, 'vweight': 1.0}

for key in list(config.sections['GROUPS'].group_table.keys()):
    if not path.isdir(config.sections['PATH'].datapath+'/'+key):
        config.sections['GROUPS'].group_table.pop(key)
    
# create a fitsnap object

if parallel:
    comm.Barrier()
snap = FitSnap()
if rank==0:
    print('second snap object made')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp

# tell ParallelTools not to check for existing fitsnap objects
pt.check_fitsnap_exist = False

if parallel:
    comm.Barrier()
snap.scraper.scrape_groups()
if parallel:
    comm.Barrier()
if rank==0:
    print('2nd scrape_groups() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.scraper.divvy_up_configs()
if parallel:
    comm.Barrier()
if rank==0:
    print('2nd divvy_up_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.data = snap.scraper.scrape_configs()
if parallel:
    comm.Barrier()
if rank==0:
    print('2nd scrape_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
snap.calculator.shared_index=0
snap.calculator.distributed_index=0
if parallel:
    comm.Barrier()
snap.process_configs()
print('rank', rank, 'finished_process_configs')
if parallel:
    comm.Barrier()
if rank==0:
    print('2nd process_configs() done')
    current_timestamp =datetime.now()
    print(current_timestamp - last_timestamp)
    last_timestamp = current_timestamp
    error_log_list = []
    structures_chosen_list = []
    for n_loop in range(AL_settings.number_of_iterations):
        snap.solver.perform_fit()
        print('loop fit', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp
        snap.solver.fit_gather()
        print('loop fit_gather', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp
        snap.solver.errors = [] # this doesn't get cleared and will cause an error when fitsnap tries to append a dictionary onto it
        # TODO: should generally check the code for other places where things get appended instead 
        #       of overwritten when called multiple times in library mode
        snap.solver.error_analysis()
        print('loop error_analysis', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp
        print(snap.solver.errors.loc['*ALL'])
        error_log_list.append(snap.solver.errors)

        if len(unlabeled_df)==0: #have fully exhausted the unlabeled pool
            break
        C = snap.solver.cov
        #A = unlabeled_df[num_col].to_numpy()
        A = pt.shared_arrays['a_copy'].array[mask_of_still_unused]
        diag = (A.dot(C) * A).sum(-1)

        # TODO: make a metadata df for objective function calculations? Can leave out the a matrix 
        #       and just return indices - always represent with E of structure

        unlabeled_df['uncertainty'] = diag
        print('loop uncertainty calculation', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp

        # this get's activated in io/outputs/output.py so we need to deactivate it
        logging.getLogger('matplotlib.ticker').disabled = True
        logging.getLogger('matplotlib.font_manager').disabled = True
        
        # if plotting, plot the correlation between errors (if known) and sqrt(uncertainty)
        if plot_stuff:
            # this only makes sense if you actually have the truth values in your 'unlabeled pool'
            if known_truth_for_unlabeled:
                preds = np.dot(A,snap.solver.fit)
                truths = pt.shared_arrays['b_copy'].array[mask_of_still_unused]
                errors = truths - preds
                plt.figure()
                plt.scatter(abs(errors), np.sqrt(diag))
                plt.ylabel('sqrt(prediction variance)')
                plt.xlabel('abs error')
                plt.title('Active Learning Step ' + str(n_loop))
                plt.savefig('uncertainty_abs_error_correlation_step_' + str(n_loop)  + '.png')
                plt.close()
                plt.figure()
                ax = plt.gca()
                plt.scatter(abs(errors), np.sqrt(diag))
                plt.ylabel('sqrt(prediction variance)')
                plt.xlabel('abs error')
                plt.title('Active Learning Step ' + str(n_loop))
                ax.set_xscale("log")
                ax.set_yscale("log")
                plt.savefig('loglog_uncertainty_abs_error_correlation_step_' + str(n_loop)  + '.png')
                plt.close()
                print('loop uncertainty error correlation plotting', n_loop, 'done')
                current_timestamp =datetime.now()
                print(current_timestamp - last_timestamp)
                last_timestamp = current_timestamp
                
        # implement the objective function here to pick some structures - ID by group and config

        ranked_structures = objective_function(unlabeled_df, EFS_reweighting = AL_settings.EFS_reweighting, FS_agg_functions=AL_settings.FS_agg_functions, objective_function = AL_settings.obj_function, \
                                               weight_by_relative_DFT_cost=AL_settings.weight_by_relative_DFT_cost)

        #if use_fitsnap_coeffs_to_scale_bispectrum_representation:
        #    x_vector_for_each_structure = x_vector_for_each_structure * snap.solver.fit

        print('loop structure ranking', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp    
        #dummy random selector for the moment
        #rand_int = np.random.randint(0,len(A))
        #(group, structure) = unlabeled_df.loc[rand_int][['Groups','Configs']].tolist()

        # TODO: implement the clustering subselection here
        # currently just take the top structure

        if AL_settings.cluster_structures:
            pass
        else:
            chosen_structures = ranked_structures.head(AL_settings.batch_size)
            structures_chosen_list.append(chosen_structures.index.to_list())

        cwd = getcwd()
    
        for (group, structure) in chosen_structures.index:
            # chosen structures data to add to training data
            mask_of_structure = (unlabeled_df['Groups']==group) & (unlabeled_df['Configs']==structure)
            a_to_append = pt.shared_arrays['a_copy'].array[mask_of_still_unused][mask_of_structure]
            
            
            b,w,g,c,r,ai,at = [],[],[],[],[],[],[]
            b = pt.shared_arrays['b_copy'].array[mask_of_still_unused][mask_of_structure].tolist()
            w = pt.shared_arrays['w_copy'].array[mask_of_still_unused][mask_of_structure].tolist()
            g = unlabeled_df[mask_of_structure]['Groups'].tolist()
            c = unlabeled_df[mask_of_structure]['Configs'].tolist()
            r = unlabeled_df[mask_of_structure]['Row_Type'].tolist()
            ai = unlabeled_df[mask_of_structure]['Atom_I'].tolist()
            at = unlabeled_df[mask_of_structure]['Atom_Type'].tolist()
            #b,p,w,g,c,r,ai,t,at = [],[],[],[],[],[],[],[],[]
            #b,p,w,g,c,r,ai,t,at = [unlabeled_df[mask_of_structure][x].tolist() for x in not_num_col]


            input_json_path = AL_settings.unlabeled_path + '/'.join([group] + [structure])
            # could store the first snap.data and pull from it, but that is a list and I would need 
            # to store the indices corresponding to each structure
            with open(input_json_path, 'r') as json_file:
                if json_file.readline()[0] == '{': #skip past comment line if it exists, otherwise start from beginning
                    json_file.seek(0)
                j = json.loads(json_file.read())
                if 'Energy' in j['Dataset']['Data'][0].keys() and j['Dataset']['Data'][0]['Energy']: #Energy recorded and not exactly 0.0 or some NaN
                    pass #we don't need to run VASP - just use the data as is
                else:
                    if VASP_class_made:
                        pass
                    else:
                        vasp_caller = VASP_runner(AL_settings)
                        VASP_class_made = True
                        FitSNAP_module_location = inspect.getabsfile(FitSnap)
                        VASP2JSON_location = '/'.join(FitSNAP_module_location.split('/')[0:-2]+['tools', 'VASP2JSON.py'])
                    vasp_output_directory = vasp_caller(input_json_path)
                    args = ['python', VASP2JSON_location, "OUTCAR", structure]
                    run(args)
                    os.rename(structure+'1.json', structure+'.json')  # name format isn't quite what we want so just renaming the file
                    # TODO: add support for doing relaxations and grabbing all the .json files (each ionic step will make one .json, sequentially numbered)
                    # TODO: put some error checking here for failed json creation
                    with open(structure+'.json', 'r') as completed_json_file:
                        if completed_json_file.readline()[0] == '{':
                            completed_json_file.seek(0)
                        cj = json.loads(completed_json_file.read())
                        if 'Energy' in cj['Dataset']['Data'][0].keys() and cj['Dataset']['Data'][0]['Energy']: #Energy recorded and not exactly 0.0 or some NaN
                            print('ERROR!! SOMETHING WENT WRONG DURING VASP CALCULATION - JSON STILL CONTAINS NO OR ZERO ENERGY TERM!!')
                        else:
                            DFT_energy = float(cj['Dataset']['Data'][0]['Energy'])
                            number_of_atoms = int(cj['Dataset']['Data'][0]['NumAtoms'])
                            DFT_forces = cj['Dataset']['Data'][0]['Forces']
                            if 'Stress' in cj['Dataset']['Data'][0].keys(): ## doing this in an if so that no stress calculations can be done as long as you aren't trying to fit stresses
                                DFT_stresses = np.array(cj['Dataset']['Data'][0]['Stress'])
                            #update b vector (currently just lammps ref model data) with DFT data
                            i=0
                            if fitting_energies:
                                b[i] += DFT_energy/number_of_atoms
                                i += 1
                            if fitting_forces:
                                b[i:i+3*len(DFT_forces)] = (np.array(DFT_forces).flatten()+np.array(b[1:1+3*len(DFT_forces)])).tolist()
                                i += number_of_atoms*3
                            if fitting_stresses:
                                b[i:i+6] = (np.array(b[i:i+6]) + np.array(DFT_stresses[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel())).tolist()

            
            #update mask that keeps shared arrays matching unlabeled pool dataframe
            original_indices_of_mask_of_structure = unlabeled_df[mask_of_structure].index.tolist()
            for i in original_indices_of_mask_of_structure:
                mask_of_still_unused[i] = False

            #remove data from unlabeled pool
            unlabeled_df.drop(unlabeled_df[mask_of_structure].index, inplace=True)
            #unlabeled_df.reset_index(inplace=True, drop=True)

                    
            #add data to snap object
            snap.pt.shared_arrays['a'].array = np.concatenate([snap.pt.shared_arrays['a'].array, a_to_append])
            snap.pt.shared_arrays['b'].array = np.concatenate([snap.pt.shared_arrays['b'].array, b])
            if snap.config.sections['GROUPS'].smartweights: #weights get updated scrape.py normally: self.data[key] /= self.group_table[self.data['Group']]['training_size'] (or ['testing_size']) - need to adjust whole group here
                #snap.scraper.group_table['Group']['training_size'] is the data we want. training size is number of structures, not number of A matrix rows
                num_currently_in_training_group = snap.scraper.group_table[group]['training_size'] #currently always adding new structures to the training data
                training_group_mask = [e == group for e in snap.pt.fitsnap_dict['Groups']]
                snap.pt.shared_arrays['w'].array[training_group_mask] *= num_currently_in_training_group / (num_currently_in_training_group+1)
                snap.pt.shared_arrays['w'].array = np.concatenate([snap.pt.shared_arrays['w'].array, np.array(w)/(num_currently_in_training_group+1)])
            else:
                snap.pt.shared_arrays['w'].array = np.concatenate([snap.pt.shared_arrays['w'].array, w]) 
            snap.pt.fitsnap_dict['Groups'].extend(g)
            snap.pt.fitsnap_dict['Configs'].extend(c)
            snap.pt.fitsnap_dict['Row_Type'].extend(r)
            snap.pt.fitsnap_dict['Atom_I'].extend(ai)
            snap.pt.fitsnap_dict['Testing'].extend([False]*len(c))
            snap.pt.fitsnap_dict['Atom_Type'].extend(at)

    
            ##unsure if I need to mess with these - I think only used during the process_configs() call. If not changing the fit or error_analysis step, I think can safely be ignored
            #snap.pt.shared_arrays['configs_per_group']
            #snap.pt.shared_arrays['number_of_atoms']
            #snap.pt.shared_arrays['number_of_dgrad_rows']
            #snap.pt.shared_arrays['ref']
        print('loop data movement', n_loop, 'done')
        current_timestamp =datetime.now()
        print(current_timestamp - last_timestamp)
        last_timestamp = current_timestamp

        print('completed loop:', n_loop)

if parallel:
    comm.Barrier()

if rank==0:
    with open('structures_chosen.dat', 'w') as f:
        for i in range(len(structures_chosen_list)):
            f.write(str(i)+  ' : ' + ', '.join('/'.join(groupconfig) for groupconfig in structures_chosen_list[i]) + '\n')

plot_stuff = True
if plot_stuff:
    if rank==0:
        for metric in ['mae', 'rmse']:
            for ind in error_log_list[-1].loc['testing_json_group', 'Unweighted', 'Testing'].index:  #'Energy', 'Force', 'Stress'
                x = [d.loc['*ALL', 'Unweighted', 'Training', ind]['ncount'] for d in error_log_list]
                y = [d.loc['testing_json_group', 'Unweighted', 'Testing', ind][metric] for d in error_log_list]
                plt.figure()
                plt.loglog(x,y, color='blue', label='Testing', marker='o',markersize=10)
                y = [d.loc['*ALL', 'Unweighted', 'Training', ind][metric] for d in error_log_list]
                plt.loglog(x,y, color='dodgerblue', label='Training', marker='o',markersize=10)
                plt.ylabel(metric)
                plt.xlabel('# of training datapoints of same type')
                plt.title(ind)
                plt.legend()
                plt.savefig('convergence_'+ind+'_'+metric+'.png')
                plt.close()
                
#plot_stuff = False
#if plot_stuff:
#    if rank==0:
#        EFS_used = []
#        if snap.config.sections['CALCULATOR'].energy:
#            EFS_used.append('Energy')
#        if snap.config.sections['CALCULATOR'].force:
#            EFS_used.append('Force')
#        if snap.config.sections['CALCULATOR'].stress:
#            EFS_used.append('Stress')
#        for quantity_type in EFS_used:
#            ncounts = [d.loc['testing_json_group', 'Unweighted', quantity_type]['ncount'] for d in error_log_list]
#            for error_metric in ['mae', 'rmse']:
#                error_metrics =  [d.loc['*ALL', 'Unweighted', quantity_type][error_metric] for d in error_log_list]
                

if parallel:
    comm.Barrier()
    
