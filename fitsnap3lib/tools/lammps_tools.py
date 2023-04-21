import os
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class LammpsTools():
    """
    Class containing methods that help calculate errors with LAMMPS potentials.

    Attributes
    ----------

    pairstyle: str
        string containing pair style info

    input_script: str
        filename/location of FitSNAP input script
    
    test_dir: str
        filename/location of test directory to read test data from
    """
    def __init__(self, pairstyle, input_script, test_dir):

        self.pairstyle = pairstyle
        self.input_script = input_script
        self.test_dir = test_dir

        self.pt = ParallelTools()
        self.pt.check_fitsnap_exist = False
        self.config = Config(arguments_lst = [input_script, "--overwrite"])
        self.config.sections["PATH"].datapath = test_dir
        keylist = os.listdir(self.config.sections["PATH"].datapath)
        test_dict = {}
        for key in keylist:
            # need to set train_size to 1.0 else FitSNAP will not actually store these configs
            # it is unintuitive but okay for now 
            test_dict[key] = {'training_size': 1.0, 'testing_size': 0.0}
        self.config.sections["GROUPS"].group_table = test_dict
        from fitsnap3lib.fitsnap import FitSnap
        self.snap = FitSnap()

        # scrape the configs

        self.snap.scrape_configs()

    def calc_mae(self, arr1, arr2):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        abs_diff = np.abs(arr1-arr2)
        mae = np.mean(abs_diff)
        return mae

    def calculate(self):
        from fitsnap3lib.calculators.lammps_snap import LammpsSnap
        calc = LammpsSnap(name='LAMMPSSNAP')
        calc.shared_index = self.snap.calculator.shared_index

        energies_all = []
        forces_all = []
        energies_test_all = []
        forces_test_all = []
        groups_all = []
        groups_forces_all = []
        for i, configuration in enumerate(self.snap.data):
            #print(i)
            calc._data = configuration
            calc._i = i
            calc._initialize_lammps() # starts a LAMMPS instance
            # set atom style, box, and make atoms
            # this function also clears the previous lammps settings and run
            calc._set_structure()
            # set neighlist
            calc._set_neighbor_list()
            # set your desired pair style
            #for pair_command in pairstyle:
            #    calc._lmp.command(pair_command)
            calc._lmp.commands_string(self.pairstyle)
            # run lammps to calculate forces and energies (add a compute for stress if desired)
            calc._lmp.command("compute PE all pe")
            calc._run_lammps()
            num_atoms = calc._data["NumAtoms"]
            lmp_atom_ids = calc._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
            # get forces and energies from lammps
            forces = calc._lmp.numpy.extract_atom("f") # Nx3 array
            energy = calc._lmp.numpy.extract_compute("PE",0,0)
            # get forces and energies from test set
            energy_test = calc._data["Energy"] 
            forces_test = calc._data["Forces"] # Nx3 array

            # format forces

            forces = forces.flatten()
            forces_test = forces_test.flatten()

            # append forces and energies to total list
            forces_all.append(forces)
            energies_all.append(energy)
            forces_test_all.append(forces_test)
            energies_test_all.append(energy_test)

            # append groups to help identify energy and force quantities
            groups_all.append(calc._data["Group"])
            groups_forces_all.extend(num_atoms*3*[calc._data["Group"]])

        #print(groups_all)
        #print(groups_forces_all)

        forces_all = np.concatenate(forces_all)
        forces_test_all = np.concatenate(forces_test_all)

        # get unique groups

        unique_groups = sorted(list(set(groups_all)))

        # calculate energy errors

        errors = {}
        for group in unique_groups:
            group_mask = [x==group for x in groups_all]
            #energies = energies_all[group_mask]
            preds = list(compress(energies_all, group_mask))
            truths = list(compress(energies_test_all, group_mask))
            mae_energy = self.calc_mae(preds, truths)

            errors[group] = {"mae_energy": mae_energy}

        # calculate force errors

        for group in unique_groups:
            group_mask = [x==group for x in groups_forces_all]
            #energies = energies_all[group_mask]
            preds = list(compress(forces_all, group_mask))
            truths = list(compress(forces_test_all, group_mask))

            mae_forces = self.calc_mae(preds, truths)

            errors[group]["mae_force"]=mae_forces

        return errors