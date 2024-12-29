from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np

import json, re
import numpy as np
from pprint import pprint

class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {'Charges': [0.0, 0.0, 0.0]} # FIXME
        self._i = 0
        self._lmp = None
        self.pt.check_lammps()

        with open(self.config.sections['REAXFF'].force_field, 'r') as file:
            self.force_field_string = file.read()

        lines = self.force_field_string.splitlines()
        line_index = int(lines[1].split()[0])+6
        number_of_elements = int(lines[line_index-4].split()[0])
        self.elements = [lines[i].split()[0] for i in range(line_index, line_index+4*number_of_elements, 4)]
        self.masses = [float(lines[i].split()[3]) for i in range(line_index, line_index+4*number_of_elements, 4)]
        self.parameters = self.config.sections['REAXFF'].parameters


    def change_parameter(self, block, atoms, name, value):

        if( block == 'ATM' ):
            num_atoms = 1
            parameters_list = [
                'r_s', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma', 'r_pi', 'valency_e',
                'alpha', 'gamma_w', 'valency_boc', 'p_ovun5', '', 'chi', 'eta', 'p_hbond', 
                'r_pi_pi', 'p_lp2', '', 'b_o_131', 'b_o_132', 'b_o_133', 'bcut_acks2', '', 
                'p_ovun2', 'p_val3', '', 'valency_val', 'p_val5', 'rcore2', 'ecore2', 'acore2']

        elif( block == 'BND' ):
            num_atoms = 2
            parameters_list = [
                'De_s','De_p','De_pp','p_be1','p_bo5','v13cor','p_bo6','p_ovun1', 
                'p_be2','p_bo3','p_bo4','','p_bo1','p_bo2','ovc','']

        elif( block == 'OFD' ):
            num_atoms = 2
            parameters_list = [
                'D', 'r_vdW', 'alpha', 'r_s', 'r_p', 'r_pp']

        elif( block == 'ANG' ):
            num_atoms = 3
            parameters_list = [
                'theta_00', 'p_val1', 'p_val2', 'p_coa1', 'p_val7', 'p_pen1', 'p_val4']

        elif( block == 'TOR' ):
            num_atoms = 4
            parameters_list = [
                'V1', 'V2', 'V3', 'p_tor1', 'p_cot1', '', '']

        elif( block == 'HBD' ):
            num_atoms = 3
            parameters_list = [
                'r0_hb', 'p_hb1', 'p_hb2', 'p_hb3']

        else:
            raise Exception(f"Block {block} not recognized, possible values are ATM, BND, OFD, ANG, TOR, HBD.")

        if( num_atoms != len(atoms) ): 
            raise Exception(f"Block {block} expected {num_atoms} atoms, but {atoms} has {len(atoms)}.")
        
        # Raises a ValueError if name not found
        parameter_index = parameters_list.index(name)

        if( block == 'ATM' ):
            atoms_string = ''.join([' {:2}'.format(atoms[0]) for a in atoms])
            extra_indent = '\n   '
        else:
            atoms_string = ''.join([' {:2d}'.format(self.elements.index(a)+1) for a in atoms])
            extra_indent = '\n      '
            
        pattern = fr'^{atoms_string}(?:\s+\-?[0-9]+\.[0-9]+){{{len(parameters_list)}}}\n'


        if( not (match := re.search(pattern, self.force_field_string, flags=re.MULTILINE|re.DOTALL)) ):
            print("pattern...", pattern)
            print("self.force_field_string...", self.force_field_string)
            raise Exception("Unable to match text to replace")

        tokens = match.group(0).split()
        tokens[num_atoms+parameter_index] = value
        tokens_formatted = [' {:8.4f}'.format(float(t)) for t in tokens[num_atoms:]]

        if( len(parameters_list)>8 ): tokens_formatted.insert(8, extra_indent)
        if( len(parameters_list)>16 ): tokens_formatted.insert(17, extra_indent)
        if( len(parameters_list)>24 ): tokens_formatted.insert(26, extra_indent)
        replacement = atoms_string + ''.join(tokens_formatted) + '\n'
        #print(replacement)
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

