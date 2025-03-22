from fitsnap3lib.io.sections.sections import Section
import re

class Reaxff(Section):

    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------

    def _parse_parameters(self, config_parameters):

        self.parameters = []
        self.values = []
        self.parameter_names = config_parameters.split()
        for p in self.parameter_names:

            # BND.H.O.p_be2
            print(p, end=' ')
            tokens = p.split('.')
            p_block = tokens.pop(0)
            p_block_index = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'].index(p_block)
            _, parameters_list = self.num_atoms_parameters_list(p_block)
            p_name = tokens.pop(-1).lower()
            p_name_index = parameters_list.index(p_name)
            p_atom_types = [self.type_mapping[a] for a in tokens]
            parameter = [p_block_index] + p_atom_types + [p_name_index]
            print(parameter, end=' ')
            self.parameters.append(parameter)
            value = self.parameter_value(p_block, tokens, p_name)
            print(value)
            self.values.append(value)

    # ----------------------------------------------------------------

    def num_atoms_parameters_list(self, block):

        if block == 'GEN':
            return 0, ['']*34 + ['bond_softness'] + ['']*7

        elif block == 'ATM':
            return 1, list(map(str.lower, [
                'r_s', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma', 'r_pi', 'valency_e',
                'alpha', 'gamma_w', 'valency_boc', 'p_ovun5', 'gauss_exp', 'chi', 'eta', 'p_hbond', 
                'r_pi_pi', 'p_lp2', '', 'b_o_131', 'b_o_132', 'b_o_133', 'bcut_acks2', '', 
                'p_ovun2', 'p_val3', '', 'valency_val', 'p_val5', 'rcore2', 'ecore2', 'acore2']))

        elif block == 'BND':
            return 2, list(map(str.lower, [
                'De_s','De_p','De_pp','p_be1','p_bo5','v13cor','p_bo6','p_ovun1',
                'p_be2','p_bo3','p_bo4','','p_bo1','p_bo2','ovc','']))

        elif block == 'OFD':
            return 2, list(map(str.lower, ['D', 'r_vdW', 'alpha', 'r_s', 'r_p', 'r_pp']))

        elif block == 'ANG':
            return 3, list(map(str.lower, ['theta_00', 'p_val1', 'p_val2', 'p_coa1', 'p_val7', 'p_pen1', 'p_val4']))

        elif block == 'TOR':
            return 4, list(map(str.lower, ['V1', 'V2', 'V3', 'p_tor1', 'p_cot1', '', '']))

        elif block == 'HBD':
            return 3, list(map(str.lower, ['r0_hb', 'p_hb1', 'p_hb2', 'p_hb3']))

        else:
            raise Exception(f"Block {block} not recognized, possible values are GEN, ATM, BND, OFD, ANG, TOR, HBD.")

    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------

    def parameter_value(self, block, atoms, name):

        if block == 'GEN':
            if name == 'bond_softness':
                return float(self.potential_string.splitlines()[36].lstrip().split(' !')[0])
            else:
                raise NotImplementedError(f"GEN.{name} not implemented.")

        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        parameter_index = parameters_list.index(name)
        return float(self.parameter_block(block, atoms).split()[num_atoms+parameter_index])


    # ----------------------------------------------------------------

    def change_general_parameter(self, name_index, value):

        # GEN.bond_softness
        if name_index == 34:
            potential_string_lines = self.potential_string.splitlines(True)
            potential_string_lines[name_index+2] = ' {:8.4f} ! GEN.bond_softness\n'.format(value)
            self.potential_string = ''.join(potential_string_lines)
        else:
            raise NotImplementedError(f"GEN.{name_index} not implemented.")

    # ----------------------------------------------------------------

    def change_parameter_string(self, block_index, atom_types, name_index, value):

        block = ['GEN','ATM','BND','OFD','ANG','TOR','HBD'][block_index]
        num_atoms, parameters_list = self.num_atoms_parameters_list(block)
        atoms = [self.elements[t-1] for t in atom_types]
        parameter_block = self.parameter_block(block, atoms)
        tokens = parameter_block.split()
        atoms_formatted = [' {:>2}'.format(a) for a in tokens[:num_atoms]]
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

    # ----------------------------------------------------------------

    def change_parameters_string(self, x):

        for p, v in zip(self.parameters, x):
            if p[0]==0:
                self.change_general_parameter(p[-1], v)
            else:
                self.change_parameter_string(p[0], p[1:-1], p[-1], v)

        return self.potential_string


