from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np

import json, re
import numpy as np
from pprint import pprint

class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {'Charges': [0.0, 0.0, 0.0]}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

        with open(self.config.sections['REAXFF'].force_field, 'r') as file:
            self.force_field_string = file.read()

        lines = self.force_field_string.splitlines()
        line_index = int(lines[1].split()[0])+6
        number_of_elements = int(lines[line_index-4].split()[0])
        self.elements = [lines[i].split()[0] for i in range(line_index, line_index+4*number_of_elements, 4)]
        self.masses = [float(lines[i].split()[3]) for i in range(line_index, line_index+4*number_of_elements, 4)]
        #print(self.elements)
        #print(self.force_field_string)
        self.parameters = self.config.sections['REAXFF'].parameters
        #pprint(self.parameters,width=150)


    def change_parameter(self, block, atoms, name, value):

        bnd_parameters = ['De_s','De_p','De_pp','p_be1','p_bo5','v13cor','p_bo6','p_ovun1',
                  'p_be2','p_bo3','p_bo4','','p_bo1','p_bo2','ovc','']

        parameter_index = bnd_parameters.index(name)
        pattern = r'^  1  2(?:\s+\-?[0-9]+\.[0-9]+){16}'
        match = re.search(pattern, self.force_field_string, flags=re.MULTILINE|re.DOTALL)
        tokens = match.group(0).split()
        tokens[parameter_index+2] = value
    
        replacement = ' {:2d} {:2d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}\n       {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(*map(int,tokens[0:2]), *map(float,tokens[2:]))

        self.force_field_string = self.force_field_string.replace(match.group(0),replacement)


    def change_parameters(self, x):

        for i in range(len(x)):
            p = self.parameters[i]
            self.change_parameter(p['block'], p['atoms'], p['name'], x[i])

    # a array is for per-atom quantities in all configs (eg charge, ...)
    # b array is for per-config quantities like energy
    # c matrix is for per-atom 3-vectors like position and velocity.

    def _prepare_lammps(self):
        self._set_structure()
        #self._set_computes()
        #self._set_neighbor_list()
        #self._lmp.command("dump 1 all custom 1 lammps.dump id x y z q")
        self._lmp.command("thermo 1")
        self._lmp.command("thermo_style custom step temp pe ke etotal press")
        self._lmp.command("pair_style reaxff NULL")

        try:
            self._lmp.pair_coeff_reaxff(self.force_field_string, self.elements)
        except:
            print('ff=',self.force_field_string)

        self._lmp.command("fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff") # maxiter 400


    def _set_box(self):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),(ay, by, cy),(az, bz, cz)) = self._data["Lattice"]
        self._lmp.command(f'region box block {-ax} {ax} {-by} {by} {-cz} {cz}')
        numtypes=self.config.sections['REAXFF'].numtypes
        self._lmp.command(f"create_box {numtypes} box")


    def _create_atoms(self):
        self._lmp.commands_list([f'mass {i+1} {self.masses[i]}' for i in range(len(self.masses))])
        self._create_atoms_helper(type_mapping=self.config.sections["REAXFF"].type_mapping)


    def _set_computes(self):
        pass


    def _create_charge(self):
        pass


    def _collect_lammps_preprocess(self):
        # Pre-process LAMMPS data by collecting data needed to allocate shared arrays.
        print("_collect_lammps_preprocess(self)")

    def process_all_configs(self,data):

        #pprint(data)

        for i, c in enumerate(data):
            self.process_configs(c, i)


    def _collect_lammps(self):

        if self.config.sections["CALCULATOR"].energy:
            config_energy = _extract_compute_np(self._lmp, "thermo_pe", 0, 0)
            self.pt.shared_arrays['b'].array[self._i] = config_energy
            #print("_collect_lammps(self)...", self._i, self._data["Energy"], config_energy)

