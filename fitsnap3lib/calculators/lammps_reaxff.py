from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, re, sys
import numpy as np
from functools import reduce
from itertools import chain, groupby
from pprint import pprint

from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
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

        reference = self.config.sections["REFERENCE"]
        if reference.units != "real" or reference.atom_style != "charge":
            raise NotImplementedError("FitSNAP-ReaxFF only supports 'units real' and 'atom_style charge'.")
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
        if self.dipole: self._lmp.command("compute dipole all dipole")


    def set_data_index(self, data_index):

        self._data_index = data_index
        self._data = self.pt.fitsnap_dict["Data"][data_index]
        self._lmp.command("delete_atoms group all")
        self._create_atoms_helper(type_mapping=self.type_mapping)


    def process_data_for_parameter_values(self, i, xi):

        try:
            self._lmp.set_reaxff_parameters(self.parameters, xi)
            self._lmp.command("run 0 post no")
            self._collect_lammps(i)
        except Exception as e:
            raise e


    def _collect_lammps(self, i):

        #dist = self._lmp.numpy.extract_compute('dist',LMP_STYLE_LOCAL,LMP_TYPE_VECTOR)
        #q = self._lmp.numpy.extract_compute('charge',LMP_STYLE_ATOM,LMP_TYPE_VECTOR)

        if self.energy:
            pe = _extract_compute_np(self._lmp, 'thermo_pe',LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            self.pt.shared_arrays['predicted_energy'].array[i][self._data_index] = pe

        if self.force:
            f = self._lmp.extract_atom('f',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
            print(f"f {f}")
            self._data['predicted_forces'] = f

        if self.dipole:
            dipole = _extract_compute_np(self._lmp,'dipole',LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR)
            self._data['predicted_dipole'] = dipole
            #print(f"dipole {dipole}")


    def allocate_per_config(self, data: list):
        """
        Allocate shared arrays for REAXFF fitting

        Args:
            data: List of data dictionaries.
        """

        # -------- DATA DISTRIBUTED LIST --------
        #pprint(data)
        self.pt.add_2_fitsnap("Data", DistributedList(len(data)))
        self.pt.fitsnap_dict["Data"] = data
        self.pt.all_barrier()
        self.pt.gather_fitsnap("Data")

        all_data = self.pt.fitsnap_dict["Data"] = list(chain.from_iterable(self.pt.fitsnap_dict["Data"]))
        len_all_data = len(all_data)
        print(f"self.pt.get_rank() {self.pt.get_rank()} len_all_data {len_all_data}")
        #if(self.pt._rank==0): pprint(all_data)


        # -------- SHARED ARRAYS --------

        print(f"self.pt.fitsnap_dict {self.pt.fitsnap_dict}")

        if self.energy:
            self.pt.create_shared_array('ground_index', len_all_data, 1, dtype='i')
            self.pt.create_shared_array('reference_energy', len_all_data, 1)
            self.pt.create_shared_array('weights', len_all_data, 1)
            popsize = self.config.sections['SOLVER'].popsize
            self.pt.create_shared_array('predicted_energy', popsize, len_all_data)

        #data = sorted(data, key=keyfunc)

        i=0

        for k, g in groupby(all_data, lambda d: d["Group"]):
            group=list(g)      # Store group iterator as a list
            print(f"k {k} g {g}")

            ground_index = 0
            ground_energy = 999999.99
            reference_energy = self.pt.shared_arrays['reference_energy'].array

            for j, d in enumerate(group):

                # FIXME: let users choose manual weights
                #if "Weight" not in d: d["Weight"] = 1.0

                if ground_energy > d["Energy"]:
                    ground_index, ground_energy = j, d["Energy"]

            for j, d in enumerate(group):
                self.pt.shared_arrays['ground_index'].array[i+j] = i + ground_index
                reference_energy[i+j] = d["Energy"] - ground_energy

            qm_y = self.pt.shared_arrays['reference_energy'].array[i:i+len(group)]
            weights = self.pt.shared_arrays['weights'].array
            weights[i:i+len(group)] = np.square(np.max(qm_y)*1.1-np.array(qm_y))
            i+=len(group)

        #if self.force: self.pt.create_shared_array('predicted_dipole', len_all_data, 1)
        if self.stress: raise NotImplementedError("FitSNAP-ReaxFF does not support stress fitting.")
        #if self.dipole: self.pt.create_shared_array('predicted_dipole', len_all_data, 1)


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
            potential_string_lines[name_index+2] = ' {:8.4f} ! GEN.bond_softness\n'.format(value)
            self.potential_string = ''.join(potential_string_lines)
        else:
            raise NotImplementedError(f"GEN.{name_index} not implemented.")


    def change_parameter_string(self, block_index, atom_types, name_index, value):

        block = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'][block_index]
        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        atoms = [self.elements[t-1] for t in atom_types]
        parameter_block = self.parameter_block(block, atoms)
        tokens = parameter_block.split()
        #tokens[num_atoms+name_index] = ' {:8.4f}'.format(value)
        tokens[num_atoms+name_index] = value
        tokens_formatted = [' {:8.4f}'.format(float(v)) for v in tokens[num_atoms:]]
        extra_indent = '\n   ' if block == 'ATM' else '\n      '
        if( len(parameters_list)>8 ): tokens_formatted.insert(8, extra_indent)
        if( len(parameters_list)>16 ): tokens_formatted.insert(17, extra_indent)
        if( len(parameters_list)>24 ): tokens_formatted.insert(26, extra_indent)
        replacement = ' ' + ' '.join(tokens[:num_atoms]) + ' ' + ''.join(tokens_formatted)
        #print(replacement)
        self.potential_string = self.potential_string.replace(parameter_block,replacement)


    def change_parameters_string(self, x):

        for p, v in zip(self.parameters, x):

            if p[0]==0:
                self.change_general_parameter(p[-1], v)
            else:
                self.change_parameter_string(p[0], p[1:-1], p[-1], v)

        return self.potential_string


