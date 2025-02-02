from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, re, sys
import numpy as np
from functools import reduce
from itertools import chain
from pprint import pprint

from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

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
        super()._initialize_lammps(printlammps=printlammps)
        self._lmp.command("clear")
        self._lmp.command("boundary p p p")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")

        # FIXME
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
        self._lmp.command("compute dist all pair/local dist")
        self._lmp.command("compute charge all property/atom q")
        self._lmp.command("compute dipole all dipole")


    def process_configs(self, data, i):

        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._lmp.set_reaxff_parameters(self.parameters, self.values )
            #self._run_lammps()
            self._lmp.command("run 1000 post no")
            self._collect_lammps()
            self._lmp = self.pt.close_lammps()
        except Exception as e:
            raise e

    def _prepare_lammps(self):

        self._lmp.command("delete_atoms group all")
        self._create_atoms_helper(type_mapping=self.type_mapping)


    def change_parameters(self, x):

        self.values = x
        #self._lmp.set_reaxff_parameters(self.parameters, x)


    def _collect_lammps(self):

        if self.energy:
            predicted_energy = _extract_compute_np(self._lmp, "thermo_pe", 0, 0)
            if np.isnan(predicted_energy):
              rounded_values = [round(float(v),2) for v in self.values]
              print(f'predicted_energy is nan {self._i} {rounded_values}')
              predicted_energy = 99e99
            self._data['predicted_energy'] = predicted_energy

            dist = self._lmp.numpy.extract_compute('dist',LMP_STYLE_LOCAL,LMP_TYPE_VECTOR)
            q = self._lmp.numpy.extract_compute('charge',LMP_STYLE_ATOM,LMP_TYPE_VECTOR)
            dipole = self._lmp.numpy.extract_compute('dipole',LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            pe = self._lmp.numpy.extract_compute('thermo_pe',LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            print(f"dist {dist[0]:.2f} dipole {dipole:.8f} q0 {q[0]: .8f} q[1] {q[1]: .8f} pe {pe: .8f}",
              end='=======================\n' if np.isnan(pe) else '\n')

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

        if block == 'GEN':
            return 0, ['']*34 + ['bond_softness'] + ['']*7

        elif block == 'ATM':
            return 1, [
                'r_s', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma', 'r_pi', 'valency_e',
                'alpha', 'gamma_w', 'valency_boc', 'p_ovun5', 'gauss_exp', 'chi', 'eta', 'p_hbond', 
                'r_pi_pi', 'p_lp2', '', 'b_o_131', 'b_o_132', 'b_o_133', 'bcut_acks2', '', 
                'p_ovun2', 'p_val3', '', 'valency_val', 'p_val5', 'rcore2', 'ecore2', 'acore2']

        elif block == 'BND':
            return 2, [
                'De_s','De_p','De_pp','p_be1','p_bo5','v13cor','p_bo6','p_ovun1',
                'p_be2','p_bo3','p_bo4','','p_bo1','p_bo2','ovc','']

        elif block == 'OFD':
            return 2, ['D', 'r_vdW', 'alpha', 'r_s', 'r_p', 'r_pp']

        elif block == 'ANG':
            return 3, ['theta_00', 'p_val1', 'p_val2', 'p_coa1', 'p_val7', 'p_pen1', 'p_val4']

        elif block == 'TOR':
            return 4, ['V1', 'V2', 'V3', 'p_tor1', 'p_cot1', '', '']

        elif block == 'HBD':
            return 3, ['r0_hb', 'p_hb1', 'p_hb2', 'p_hb3']

        else:
            raise Exception(f"Block {block} not recognized, possible values are GEN, ATM, BND, OFD, ANG, TOR, HBD.")


    def parameter_block(self, block, atoms):

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)

        if( num_atoms != len(atoms) ):
            raise Exception(f"Block {block} expected {num_atoms} atoms, but {atoms} has {len(atoms)}.")
        
        if block == 'ATM':
            atoms_string = r'\s*' + atoms[0]
            extra_indent = '\n   '
        else:
            atoms_string = ''.join([r'\s*'+str(self.elements.index(a)+1) for a in atoms])
            extra_indent = '\n      '

        pattern = fr'^{atoms_string}(?:\s+\-?[0-9]+\.[0-9]+){{{len(parameters_list)}}}'

        if( not (match := re.search(pattern, self.potential_string, flags=re.MULTILINE|re.DOTALL)) ):
            print("pattern...", pattern)
            print(f"--------\nself.potential_string...\n{self.potential_string}\n--------\n")
            raise Exception("Unable to match text to replace")

        return match.group(0)


    def parameter_value(self, block, atoms, name):

        if block == 'GEN':
            if name == 'bond_softness':
                return float(self.potential_string.splitlines()[36].lstrip().split(' !')[0])
            else:
                raise NotImplementedError(f"GEN.{name} not implemented.")

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        parameter_index = parameters_list.index(name)
        return float(self.parameter_block(block, atoms).split()[num_atoms+parameter_index])


    def _parse_parameters(self, config_parameters):

        self.parameters = []
        self.values = []

        for p in config_parameters.split():

            # BND.H.O.p_be2
            print(p, end=' ')
            tokens = p.split('.')
            p_block = tokens.pop(0)
            p_block_index = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'].index(p_block)
            _, parameters_list = self.num_atoms_parameters_list(p_block)
            p_name = tokens.pop(-1)
            p_name_index = parameters_list.index(p_name)
            p_atom_types = [self.type_mapping[a] for a in tokens]
            parameter = [p_block_index] + p_atom_types + [p_name_index]
            print(parameter, end=' ')
            self.parameters.append(parameter)
            value = self.parameter_value(p_block, tokens, p_name)
            print(value)
            self.values.append(value)


    def change_general_parameter(self, name_index, value):

        # GEN.bond_softness
        if name_index == 34:
            potential_string_lines = self.potential_string.splitlines(True)
            potential_string_lines[name_index+2] = ' {:12.8f} ! GEN.bond_softness'.format(value)
            self.potential_string = ''.join(potential_string_lines)
        else:
            raise NotImplementedError(f"GEN.{name_index} not implemented.")


    def change_parameter_string(self, block_index, atom_types, name_index, value):

        block = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'][block_index]
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
        replacement = ' ' + ' '.join(tokens[:num_atoms]) + ' ' + ' '.join(tokens_formatted) + '\n'
        #print(replacement)
        self.potential_string = self.potential_string.replace(parameter_block,replacement)


    def change_parameters_string(self, x):

        for p, v in zip(self.parameters, x):

            if p[0]==0:
                self.change_general_parameter(p[-1], v)
            else:
                self.change_parameter_string(p[0], p[1:-1], p[-1], v)

        return self.potential_string


