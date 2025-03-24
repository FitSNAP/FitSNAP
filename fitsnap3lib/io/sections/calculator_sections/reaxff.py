from fitsnap3lib.io.sections.sections import Section
import re
import numpy as np

class Reaxff(Section):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['potential', 'parameters']
        self.potential = self.get_value("REAXFF", "potential", "None", "str")
        self.parameters = self.get_value("REAXFF", "parameters", "None", "str")

        with open(self.potential, 'r') as file:
            self.potential_string = file.read()

        lines = self.potential_string.splitlines()
        line_index = int(lines[1].split()[0])+6
        number_of_elements = int(lines[line_index-4].split()[0])
        elements_range = range(line_index, line_index+4*number_of_elements, 4)
        self.elements = [lines[i].split()[0] for i in elements_range]
        self.masses = [float(lines[i].split()[3]) for i in elements_range]
        self.type_mapping = {e: self.elements.index(e)+1 for e in self.elements}
        self._parse_parameters(self.parameters)

        # FIXME: remove later
        # self.delete()

    # --------------------------------------------------------------------------------------------

    def _parse_parameters(self, config_parameters):

        bounds = {
            # "": (, ),       #
            # ATM

            "r_s": (0.3, 1.5),        # Covalent sigma-bond radius (Å)
            "r_pi": (0.3, 1.5),       # Covalent π-bond radius (Å)
            "r_pi_pi": (0.5, 2.0),    # π–π bond radius

            "r_vdw": (0.5, 3.5),      # van der Waals radius (Å)
            "epsilon": (0.01, 0.50),  # vdW dissociation energy (kcal/mol)
            "alpha": (0.0, 10.0),     # van der Waals alpha parameter
            "gamma_w": (0.0, 3.0),    # van der Waals shielding width

            "gamma": (0.5, 20.0),     # Valence orbital exponent for QEq/ACKS2
            "chi": (-10.0, +10.0),    # Electronegativity (ACKS2, eV)
            "eta": (0.0, 20.0),       # Hardness (ACKS2, eV)
            "bcut_acks2": (1.0, 6.0), # ACKS2 atomic softness cutoff

            "p_val3": (-3.0, 3.0),    # Valence angle penalty term
            "p_val5": (-3.0, 3.0),    # Valence angle penalty term
            "b_o_131": (-1.0, 1.0),   # BO correction term 131
            "b_o_132": (-1.0, 1.0),   # BO correction term 132
            "b_o_133": (-1.0, 1.0),   # BO correction term 133
            "p_ovun2": (-8.0, +1.0),  # Angle-based undercoordination penalty
            "p_ovun5": (-10.0, 0.0),  # Undercoordination energy penalty
            "p_lp2": (0.0, 30.0),     # Lone pair penalty, atom-specific

            "rcore2": (0.1, 2.0),     # Inner core vdW repulsion radius (Å)
            "ecore2": (0.0, 0.5),     # Inner core vdW repulsion energy (kcal/mol)
            "acore2": (0.0, 10.0),    # Inner core vdW exponential factor

            # BND
            "De_s": (50.0, 200.0),    # sigma-bond dissociation energy (kcal/mol)
            "De_p": (0.0, 150.0),     # pi-bond dissociation energy
            "De_pp": (0.0, 80.0),     # pipi-bond dissociation energy
            "p_be1": (-1.5, 0.5),     # Bond energy parameter coefficient
            "p_be2": (0.1, 10.0),     # Bond energy parameter exponent
            "p_bo1": (-1.0, 0.2),     # sigma-bond order coefficient
            "p_bo2": (1.0, 20.0),     # sigma-bond order exponent
            "p_bo3": (-1.0, 0.2),     # pi-bond order coefficient
            "p_bo4": (1.0, 20.0),     # pi-bond order exponent
            "p_bo5": (-1.0, 0.2),     # pipi-bond order coefficient
            "p_bo6": (0.0, 50.0),     # pipi-bond order exponent
            "p_ovun1": (0.0, 1.0),    # Overcoordination penalty (lowers BO when overcoord)

            # OFD

        }

        self.parameters = []
        self.values = []
        self.parameter_bounds = []
        self.parameter_names = config_parameters.split()
        for p in self.parameter_names:

            # eg. BND.H.O.p_be2
            tokens = p.split('.')
            p_block = tokens.pop(0)
            p_block_index = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'].index(p_block)
            _, parameters_list = self.num_atoms_parameters_list(p_block)
            p_name = tokens.pop(-1)
            p_name_index = parameters_list.index(p_name)
            p_atom_types = [self.type_mapping[a] for a in tokens]
            parameter = [p_block_index] + p_atom_types + [p_name_index]
            value = self.parameter_value(p_block, tokens, p_name)

            if p_name in bounds:
                p_bounds = bounds[p_name]
            else:
                delta = max(0.5 * abs(value), 1.0)
                p_bounds = (value-delta, value+delta)

            value = np.clip(value, p_bounds[0], p_bounds[1])
            #print(p, parameter, value, p_bounds)

            self.parameters.append(parameter)
            self.values.append(value)
            self.parameter_bounds.append(p_bounds)

    # --------------------------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------------------------

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

        pattern = fr'^{atoms_string}(?:\s+\-?[0-9]+\.[0-9]+){{{len(parameters_list)}}}$'

        if( not (match := re.search(pattern, self.potential_string, flags=re.MULTILINE|re.DOTALL)) ):
            print("pattern...", pattern)
            print(f"--------\nself.potential_string...\n{self.potential_string}\n--------\n")
            raise Exception("Unable to match text to replace")

        return match.group(0)

    # --------------------------------------------------------------------------------------------

    def parameter_value(self, block, atoms, name):

        if block == 'GEN':
            if name == 'bond_softness':
                return float(self.potential_string.splitlines()[36].lstrip().split(' !')[0])
            else:
                raise NotImplementedError(f"GEN.{name} not implemented.")

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        parameter_index = parameters_list.index(name)
        return float(self.parameter_block(block, atoms).split()[num_atoms+parameter_index])


    # --------------------------------------------------------------------------------------------

    def change_general_parameter(self, name_index, value):

        # GEN.bond_softness
        if name_index == 34:
            potential_string_lines = self.potential_string.splitlines(True)
            potential_string_lines[name_index+2] = ' {:8.4f} ! GEN.bond_softness\n'.format(value)
            self.potential_string = ''.join(potential_string_lines)
        else:
            raise NotImplementedError(f"GEN.{name_index} not implemented.")

    # --------------------------------------------------------------------------------------------

    def change_parameter_string(self, block_index, atom_types, name_index, value):

        block = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'][block_index]
        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        atoms = [self.elements[t-1] for t in atom_types]
        parameter_block = self.parameter_block(block, atoms)
        tokens = parameter_block.split()
        atom_format = ' {:<2}' if block=='ATM' else ' {:>2}'
        atoms_formatted = [atom_format.format(a) for a in tokens[:num_atoms]]
        #tokens[num_atoms+name_index] = ' {:8.4f}'.format(value)
        tokens[num_atoms+name_index] = value
        tokens_formatted = [' {:8.4f}'.format(float(v)) for v in tokens[num_atoms:]]
        extra_indent = '\n   ' if block == 'ATM' else '\n      '
        if( len(parameters_list)>8 ): tokens_formatted.insert(8, extra_indent)
        if( len(parameters_list)>16 ): tokens_formatted.insert(17, extra_indent)
        if( len(parameters_list)>24 ): tokens_formatted.insert(26, extra_indent)
        replacement = ''.join(atoms_formatted) + ''.join(tokens_formatted)
        #print(replacement)
        self.potential_string = self.potential_string.replace(parameter_block,replacement)

    # --------------------------------------------------------------------------------------------

    def change_parameters_string(self, x):

        for p, v in zip(self.parameters, x):
            if p[0]==0:
                self.change_general_parameter(p[-1], v)
            else:
                self.change_parameter_string(p[0], p[1:-1], p[-1], v)

        return self.potential_string

    # --------------------------------------------------------------------------------------------

