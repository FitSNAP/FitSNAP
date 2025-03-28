from fitsnap3lib.io.sections.sections import Section
import re
import numpy as np
from itertools import product, combinations_with_replacement
from pprint import pprint

class Reaxff(Section):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['potential', 'parameters']
        self.potential = self.get_value("REAXFF", "potential", "None", "str")
        with open(self.potential, 'r') as file: self.potential_string = file.read()
        lines = self.potential_string.splitlines()
        line_index = int(lines[1].split()[0])+6
        number_of_elements = int(lines[line_index-4].split()[0])
        elements_range = range(line_index, line_index+4*number_of_elements, 4)
        self.elements = [lines[i].split()[0] for i in elements_range]
        self.masses = [float(lines[i].split()[3]) for i in elements_range]
        self.type_mapping = {e: self.elements.index(e)+1 for e in self.elements}
        self.number_format = ' {:12.8f}'

        self.bounds = {

            # GEN
            "bond_softness": (200.0, 800.0),      # ACKS2 bond softness

            # ATM
            "r_s":           (0.4, 1.1),          # Covalent sigma-bond radius (Å)
            "r_pi":          (1.1, 2.0),          # Covalent π-bond radius (Å)
            "r_pi_pi":       (2.0, 3.5),          # π–π bond radius
            "r_vdw":         (0.5, 3.5),          # vdW radius (Å)
            "epsilon":       (0.01, 0.50),        # vdW dissociation energy (kcal/mol)
            "alpha":         (0.0, 10.0),         # vdW alpha parameter
            "gamma_w":       (0.0, 3.0),          # vdW shielding width

            "gamma":         (0.5, 20.0),         # Valence orbital exponent for QEq/ACKS2
            "chi":           (-10.0, 10.0),       # Electronegativity (ACKS2, eV)
            "eta":           (0.0, 20.0),         # Hardness (ACKS2, eV)
            "bcut_acks2":    (1.0, 6.0),          # ACKS2 atomic softness cutoff

            "p_val3":        (-3.0, 3.0),         # Valence angle penalty term
            "p_val5":        (-3.0, 3.0),         # Valence angle penalty term
            "b_o_131":       (-1.0, 1.0),         # BO correction term 131
            "b_o_132":       (-1.0, 1.0),         # BO correction term 132
            "b_o_133":       (-1.0, 1.0),         # BO correction term 133
            "p_ovun2":       (-8.0, 1.0),         # Angle-based undercoordination penalty
            "p_ovun5":       (-10.0, 0.0),        # Undercoordination energy penalty
            "p_lp2":         (0.0, 30.0),         # Lone pair penalty, atom-specific

            "rcore2":        (0.1, 2.0),          # Inner core vdW repulsion radius (Å)
            "ecore2":        (0.0, 0.5),          # Inner core vdW repulsion energy (kcal/mol)
            "acore2":        (0.0, 10.0),         # Inner core vdW exponential factor

            # BND
            "De_s":          (50.0, 200.0),       # sigma-bond dissociation energy (kcal/mol)
            "De_p":          (0.0, 150.0),        # pi-bond dissociation energy
            "De_pp":         (0.0, 80.0),         # pipi-bond dissociation energy
            "p_be1":         (-1.5, 0.5),         # Bond energy parameter coefficient
            "p_be2":         (1.0, 10.0),         # Bond energy parameter exponent
            "p_bo1":         (-1.0, 0.2),         # sigma-bond order coefficient
            "p_bo2":         (1.0, 20.0),         # sigma-bond order exponent
            "p_bo3":         (-1.0, 0.2),         # pi-bond order coefficient
            "p_bo4":         (1.0, 20.0),         # pi-bond order exponent
            "p_bo5":         (-1.0, 0.2),         # pipi-bond order coefficient
            "p_bo6":         (0.0, 50.0),         # pipi-bond order exponent
            "p_ovun1":       (0.8, 1.2),          # Overcoordination penalty

            # OFD
            "D":             (0.01, 0.5),         # van der Waals well depth (kcal/mol)
            "r_vdW":         (0.5, 3.5),          # van der Waals contact distance (Å)
            "alpha":         (0.0, 10.0),         # vdW decay sharpness
            "r_s":           (0.4, 1.1),          # σ-bond cutoff radius (Å)
            "r_p":           (1.1, 2.0),          # π-bond cutoff radius (Å)
            "r_pp":          (2.0, 3.5),          # π–π bond cutoff radius (Å)

            # ANG
            "theta_00":      (0.0, 180.0),        # θ_eq = 180 - theta_00 (deg)
            "p_val1":        (-1.0, 10.0),        # Angular stiffness term
            "p_val2":        (-2.0, 2.0),         # Angular curvature term
            "p_coa1":        (-1.0, 1.0),         # π-conjugation energy
            "p_val7":        (-10.0, 0.0),        # Undercoordination angular penalty
            "p_pen1":        (0.0, 100.0),        # Angular distortion penalty (kcal/mol)
            "p_val4":        (-2.0, 2.0),         # Angular modulation parameter

            # TOR
            "V1":            (0.0, 20.0),         # 1st cosine torsion barrier (kcal/mol)
            "V2":            (-10.0, 10.0),       # 2nd cosine torsion barrier
            "V3":            (-10.0, 10.0),       # 3rd cosine torsion barrier
            "p_tor1":        (0.0, 10.0),         # Torsional stiffness scaling
            "p_cot1":        (-5.0, 5.0),         # Conjugation adjustment for torsions

            # HBD
            "r0_hb":         (1.5, 2.5),          # H-bond distance (Å)
            "p_hb1":         (0.0, 20.0),         # H-bond strength (kcal/mol)
            "p_hb2":         (0.1, 5.0),          # H-bond decay with BO
            "p_hb3":         (0.0, 1.0),          # Angular penalty shaping

        }

        self._parse_parameters(self.get_value("REAXFF", "parameters", "None", "str"))

    # --------------------------------------------------------------------------------------------

    def _parse_parameters(self, config_parameters):
        self.parameters = []
        self.values = []
        self.parameter_bounds = []
        self.parameter_names = []

        existing_tuples = self._get_existing_tuples()

        expanded_parameters = self._expand_parameter_wildcards(config_parameters, existing_tuples)

        for p in expanded_parameters:
            tokens = p.split('.')
            p_block = tokens.pop(0)
            p_block_index = ['GEN', 'ATM', 'BND', 'OFD', 'ANG', 'TOR', 'HBD'].index(p_block)
            _, parameters_list = self.num_atoms_parameters_list(p_block)
            p_name = tokens.pop(-1)
            p_name_index = parameters_list.index(p_name)
            p_atom_types = [self.type_mapping[a] for a in tokens]
            parameter = [p_block_index] + p_atom_types + [p_name_index]
            value = self.parameter_value(p_block, tokens, p_name)
            if p_name in self.bounds:
                p_bounds = self.bounds[p_name]
            else:
                delta = max(0.5 * abs(value), 1.0)
                p_bounds = (value - delta, value + delta)
            warning = "" if p_bounds[0] <= value <= p_bounds[1] else "WARNING value outside bounds"
            self.pt.single_print(p, parameter, value, p_bounds, warning)

            #value = np.clip(value, p_bounds[0], p_bounds[1])
            if value<p_bounds[0] or p_bounds[1] < value: value = (p_bounds[0]+p_bounds[1])/2.0

            self.parameter_names.append(p)
            self.parameters.append(parameter)
            self.values.append(value)
            self.parameter_bounds.append(p_bounds)

    # --------------------------------------------------------------------------------------------

    def _get_existing_tuples(self):
        import re
        tuples = {'BND': set(), 'OFD': set(), 'ANG': set(), 'TOR': set(), 'HBD': set()}

        patterns = {
            'BND': (2, 8),
            'OFD': (2, 6),
            'ANG': (3, 7),
            'TOR': (4, 7),
            'HBD': (3, 4),
        }

        float_re = r'^-?\d+\.\d+(?:[Ee][-+]?\d+)?$'
        int_re = r'^\d+$'

        for line in self.potential_string.splitlines():
            tokens = line.strip().split()
            if not tokens:
                continue

            for block, (nint, nfloat) in patterns.items():
                if len(tokens) != nint + nfloat:
                    continue
                if all(re.fullmatch(int_re, tok) for tok in tokens[:nint]) and \
                   all(re.fullmatch(float_re, tok) for tok in tokens[nint:]):
                    try:
                        ids = list(map(int, tokens[:nint]))
                        elems = tuple(self.elements[i - 1] for i in ids)
                        if block == 'BND':
                            i, j = self.elements.index(elems[0]), self.elements.index(elems[1])
                            elems = (elems[0], elems[1]) if i <= j else (elems[1], elems[0])
                        tuples[block].add(elems)
                    except Exception:
                        continue

        return tuples

    # --------------------------------------------------------------------------------------------

    def _expand_parameter_wildcards(self, config_parameters, existing_tuples):
        expanded = []

        def expand_atom(a):
            return self.elements if a == '*' else [a]

        for p in config_parameters.split():
            tokens = p.split('.')
            p_block = tokens[0]
            if '*' not in tokens:
                expanded.append(p)
                continue
            name = tokens[-1]
            args = tokens[1:-1]
            if p_block == 'ATM':
                for e in self.elements:
                    expanded.append(f'ATM.{e}.{name}')
            elif p_block == 'BND':
                a1, a2 = args
                for x in expand_atom(a1):
                    for y in expand_atom(a2):
                        i, j = self.elements.index(x), self.elements.index(y)
                        a, b = (x, y) if i <= j else (y, x)
                        if (a, b) in existing_tuples['BND']:
                            expanded.append(f'BND.{a}.{b}.{name}')
            elif p_block == 'OFD':
                a1, a2 = args
                for x in expand_atom(a1):
                    for y in expand_atom(a2):
                        if (x, y) in existing_tuples['OFD']:
                            expanded.append(f'OFD.{x}.{y}.{name}')
            elif p_block in {'ANG', 'TOR', 'HBD'}:
                nargs = len(args)
                expansions = [[]]
                for a in args:
                    new_expansions = []
                    for partial in expansions:
                        for e in expand_atom(a):
                            new_expansions.append(partial + [e])
                    expansions = new_expansions
                for tup in expansions:
                    if tuple(tup) in existing_tuples[p_block]:
                        expanded.append(f'{p_block}.' + '.'.join(tup) + f'.{name}')
            else:
                raise ValueError(f"Wildcard not supported for block: {p_block}")
        return expanded








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

        if num_atoms != len(atoms):
            raise Exception(f"Block {block} expected {num_atoms} atoms, got {atoms} ({len(atoms)} atoms)")

        whitespace = r'\s+'
        float_pattern = r'-?\d+\.\d+'  # fixed-point only, no exponent

        def atom_symbol_to_index(a):
            if a == '0':
                return r'\d+'
            try:
                return str(self.elements.index(a) + 1)
            except ValueError:
                raise Exception(f"Atom type '{a}' not in self.elements: {self.elements}")

        def atoms_to_pattern(block, atoms):
            if block == 'ATM':
                return whitespace.join([a for a in atoms])
            else:
                return whitespace.join([atom_symbol_to_index(a) for a in atoms])

        atoms_string = atoms_to_pattern(block, atoms)
        pattern = fr'^\s*{atoms_string}(?:{whitespace}{float_pattern}){{{len(parameters_list)}}}\s*$'

        match = re.search(pattern, self.potential_string, flags=re.MULTILINE)

        if not match and block == 'TOR':
            wildcard_atoms = ['0' if a != '0' else '0' for a in atoms]
            atoms_string = atoms_to_pattern(block, wildcard_atoms)
            pattern = fr'^\s*{atoms_string}(?:{whitespace}{float_pattern}){{{len(parameters_list)}}}\s*$'
            match = re.search(pattern, self.potential_string, flags=re.MULTILINE)

        if not match:
            print(f"*** parameter_block failed for block {block}")
            print(f"*** atoms: {atoms}")
            print(f"*** regex: {pattern}")
            print("----- potential_string lines matching first atom ------")
            for line in self.potential_string.splitlines():
                if atoms[0] in line or atom_symbol_to_index(atoms[0]) in line:
                    print(">>>", line)
            print("--------------------------------------------------------")
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
        parameter_block = self.parameter_block(block, atoms)
        parameter_index = parameters_list.index(name)
        #print(f"*** parameter_block {parameter_block} parameter_index {parameter_index}")
        return float(parameter_block.split()[num_atoms+parameter_index])

    # --------------------------------------------------------------------------------------------

    def change_general_parameter(self, name_index, value):

        # GEN.bond_softness
        if name_index == 34:
            potential_string_lines = self.potential_string.splitlines(True)
            potential_string_lines[name_index+2] = self.number_format + \
              ' ! GEN.bond_softness\n'.format(value)
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
        atom_format = '{:<2}' if block=='ATM' else '{:>2}'
        atoms_formatted = [atom_format.format(a) for a in tokens[:num_atoms]]
        tokens[num_atoms+name_index] = value
        tokens_formatted = [self.number_format.format(float(v)) for v in tokens[num_atoms:]]
        extra_indent = '\n  ' if block == 'ATM' else '\n      '
        if( len(parameters_list)>8 ): tokens_formatted.insert(8, extra_indent)
        if( len(parameters_list)>16 ): tokens_formatted.insert(17, extra_indent)
        if( len(parameters_list)>24 ): tokens_formatted.insert(26, extra_indent)
        replacement = ' '.join(atoms_formatted) + ''.join(tokens_formatted)
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

