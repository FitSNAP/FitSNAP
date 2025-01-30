from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, re, sys
import numpy as np
from functools import reduce
from itertools import chain
from pprint import pprint

class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._i = 0
        self._lmp = None
        self.pt.check_lammps()
        self.potential_path = self.config.sections['REAXFF'].potential
        self.charge_fix = self.config.sections['CALCULATOR'].charge_fix

        with open(self.potential_path, 'r') as file:
            self.potential_string = file.read()

        lines = self.potential_string.splitlines()
        line_index = int(lines[1].split()[0])+6
        number_of_elements = int(lines[line_index-4].split()[0])
        self.elements = [lines[i].split()[0] for i in range(line_index, line_index+4*number_of_elements, 4)]
        self.masses = [float(lines[i].split()[3]) for i in range(line_index, line_index+4*number_of_elements, 4)]
        self.type_mapping = {e: self.elements.index(e)+1 for e in self.elements}
        self._parse_parameters(self.config.sections['REAXFF'].parameters)
        self._initialize_lammps()


    def __del__(self):
        self._lmp = self.pt.close_lammps()
        del self

    def _initialize_lammps(self, printlammps=0):
        super()._initialize_lammps(printlammps)
        self._lmp.command("boundary p p p")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")
        #xlo, ylo, zlo = np.min(self._data["Positions"],axis=0)-10.0
        #xhi, yhi, zhi = np.max(self._data["Positions"],axis=0)+10.0
        #print(xlo, ylo, zlo, xhi, yhi, zhi)
        #self._lmp.command(f'region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}')
        self._lmp.command(f"region box block -15 15 -15 15 -15 15")
        self._lmp.command(f"create_box {len(self.elements)} box")
        self._lmp.commands_list([f"mass {i+1} {self.masses[i]}" for i in range(len(self.masses))])
        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command(f"pair_coeff * * {self.potential_path} {' '.join(self.elements)}")
        self._lmp.command(self.charge_fix)


    def process_configs(self, data, i):

        try:
            self._data = data
            self._i = i
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps()
        except Exception as e:
            raise e


    def _prepare_lammps(self):

        self._lmp.command("delete_atoms group all")
        self._create_atoms_helper(type_mapping=self.type_mapping)


    def change_parameters(self, x):

        for i, xi in enumerate(x):
            self.parameters[i][-1] = float(xi)

        #print(self.parameters)

        self._lmp.set_reaxff_parameters(self.parameters)


    def _collect_lammps(self):

        if self.energy:
            predicted_energy = _extract_compute_np(self._lmp, "thermo_pe", 0, 0)
            self._data['predicted_energy'] = predicted_energy if not np.isnan(predicted_energy) else 9999
            #print("_collect_lammps(self)...", self._i, self._data["Energy"], config_energy)


    def allocate_per_config(self, data: list):
        """
        Allocate shared arrays for REAXFF fitting

        Args:
            data: List of data dictionaries.
        """

        self.pt.add_2_fitsnap("Data", DistributedList(len(data)))

        for i, d in enumerate(data):
            self.pt.fitsnap_dict["Data"][i] = d

        self.pt.all_barrier()
        self.pt.gather_fitsnap("Data")

        all_data = self.pt.fitsnap_dict["Data"] = list(chain.from_iterable(self.pt.fitsnap_dict["Data"]))
        print(f"self.pt.get_rank()={self.pt.get_rank()} len(all_data)={len(all_data)}")

        #if(self.pt._rank==0): pprint(all_data)

    def num_atoms_parameters_list(self, block):

        if( block == 'ATM' ):
            yield 1
            yield [
                'r_s', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma', 'r_pi', 'valency_e',
                'alpha', 'gamma_w', 'valency_boc', 'p_ovun5', '', 'chi', 'eta', 'p_hbond', 
                'r_pi_pi', 'p_lp2', '', 'b_o_131', 'b_o_132', 'b_o_133', 'bcut_acks2', '', 
                'p_ovun2', 'p_val3', '', 'valency_val', 'p_val5', 'rcore2', 'ecore2', 'acore2']

        elif( block == 'BND' ):
            yield 2
            yield [
                'De_s','De_p','De_pp','p_be1','p_bo5','v13cor','p_bo6','p_ovun1',
                'p_be2','p_bo3','p_bo4','','p_bo1','p_bo2','ovc','']

        elif( block == 'OFD' ):
            yield 2
            yield ['D', 'r_vdW', 'alpha', 'r_s', 'r_p', 'r_pp']

        elif( block == 'ANG' ):
            yield 3
            yield ['theta_00', 'p_val1', 'p_val2', 'p_coa1', 'p_val7', 'p_pen1', 'p_val4']

        elif( block == 'TOR' ):
            yield 4
            yield ['V1', 'V2', 'V3', 'p_tor1', 'p_cot1', '', '']

        elif( block == 'HBD' ):
            yield 3
            yield ['r0_hb', 'p_hb1', 'p_hb2', 'p_hb3']

        else:
            raise Exception(f"Block {block} not recognized, possible values are ATM, BND, OFD, ANG, TOR, HBD.")


    def parameter_block(self, block, atoms):

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)

        if( num_atoms != len(atoms) ):
            raise Exception(f"Block {block} expected {num_atoms} atoms, but {atoms} has {len(atoms)}.")
        
        if( block == 'ATM' ):
            atoms_string = ''.join(['\s*{:2}'.format(atoms[0]) for a in atoms])
            extra_indent = '\n   '
        else:
            atoms_string = ''.join(['\s*'+str(self.elements.index(a)+1) for a in atoms])
            extra_indent = '\n      '

        pattern = fr'^{atoms_string}(?:\s+\-?[0-9]+\.[0-9]+){{{len(parameters_list)}}}'

        if( not (match := re.search(pattern, self.potential_string, flags=re.MULTILINE|re.DOTALL)) ):
            print("pattern...", pattern)
            #print("self.potential_string...", self.potential_string)
            raise Exception("Unable to match text to replace")

        return match.group(0)


    def parameter_value(self, block, atoms, name):

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        parameter_index = parameters_list.index(name)
        return float(self.parameter_block(block, atoms).split()[num_atoms+parameter_index])


    def _parse_parameters(self, config_parameters):

        self.parameters = []

        for p in config_parameters.split():

            # BND.H.O.p_be2
            tokens = p.split('.')
            p_block = tokens.pop(0)
            p_block_index = ['ATM','BND','OFD','ANG','TOR','HBD'].index(p_block)
            _, parameters_list = self.num_atoms_parameters_list(p_block)
            p_name = tokens.pop(-1)
            p_name_index = parameters_list.index(p_name)
            p_atom_types = [self.type_mapping[a] for a in tokens]
            p_value = self.parameter_value(p_block, tokens, p_name)
            parameter = [p_block_index] + p_atom_types + [p_name_index, p_value]
            print(parameter)
            self.parameters.append(parameter)


    def change_parameter_string(self, block_index, atom_types, name_index, value):

        block = ['ATM','BND','OFD','ANG','TOR','HBD'][block_index]
        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        atoms = [self.elements[t-1] for t in atom_types]
        parameter_block = self.parameter_block(block, atoms)
        tokens = parameter_block.split()
        tokens[num_atoms+name_index] = ' {:12.8f}'.format(value)
        tokens_formatted = tokens[num_atoms:]

        extra_indent = '\n   ' if block == 'ATM' else '\n      '

        if( len(parameters_list)>8 ): tokens_formatted.insert(8, extra_indent)
        if( len(parameters_list)>16 ): tokens_formatted.insert(17, extra_indent)
        if( len(parameters_list)>24 ): tokens_formatted.insert(26, extra_indent)
        replacement = ' '.join(tokens[:num_atoms]) + ' ' + ' '.join(tokens_formatted) + '\n'
        #print(replacement)
        self.potential_string = self.potential_string.replace(parameter_block,replacement)


    def change_parameters_string(self, x):

        for i in range(len(x)):
            p = self.parameters[i]
            self.change_parameter_string(p[0], p[1:-2], p[-2], p[-1])

        return self.potential_string


