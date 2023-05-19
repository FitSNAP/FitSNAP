from fitsnap3lib.scrapers.scrape import Scraper
import numpy as np
from random import shuffle, random
from os import path, listdir
from copy import copy
import re
from _collections import defaultdict


#config = Config()
#pt = ParallelTools()


UNPROCESSED_KEYS = ['uid']

PROPERTY_NAME_MAP = {'positions': 'pos',
                     'numbers': 'Z',
                     'charges': 'charge',
                     'symbols': 'species'}

REV_PROPERTY_NAME_MAP = dict(zip(PROPERTY_NAME_MAP.values(), PROPERTY_NAME_MAP.keys()))

# all_properties = ['energy', 'forces', 'stress', 'stresses', 'dipole',
#                   'charges', 'magmom', 'magmoms', 'free_energy', 'energies']


def key_val_str_to_dict(string, sep=None):
    """
    Parse an xyz properties string in a key=value and return a dict with
    various values parsed to native types.

    Accepts brackets or quotes to delimit values. Parses integers, floats
    booleans and arrays thereof. Arrays with 9 values are converted to 3x3
    arrays with Fortran ordering.

    If sep is None, string will split on whitespace, otherwise will split
    key value pairs with the given separator.

    """
    # store the closing delimiters to match opening ones
    delimiters = {
        "'": "'",
        '"': '"',
        '(': ')',
        '{': '}',
        '[': ']',
    }

    # Make pairs and process afterwards
    # List of characters for each entry, add a new list for new value
    kv_pairs = [
        [[]]]  
    # push and pop closing delimiters
    delimiter_stack = []  
    # add escaped sequences verbatim
    escaped = False  

    # parse character-by-character unless someone can do nested brackets
    # and escape sequences in a regex
    for char in string.strip():
        # bypass everything if escaped
        if escaped:  
            kv_pairs[-1][-1].extend(['\\', char])
            escaped = False
        # inside brackets
        elif delimiter_stack:  
            # find matching delimiter
            if char == delimiter_stack[-1]:  
                delimiter_stack.pop()
            elif char in delimiters:
                # nested brackets
                delimiter_stack.append(delimiters[char])  
            elif char == '\\':
                # so escaped quotes can be ignored
                escaped = True  
            else:
                # inside quotes, add verbatim
                kv_pairs[-1][-1].append(char)  
        elif char == '\\':
            escaped = True
        elif char in delimiters:
            # brackets or quotes
            delimiter_stack.append(delimiters[char])  
        elif (sep is None and char.isspace()) or char == sep:
            # empty, beginning of string
            if kv_pairs == [[[]]]:  
                continue
            elif kv_pairs[-1][-1] == []:
                continue
            else:
                kv_pairs.append([[]])
        elif char == '=':
            if kv_pairs[-1] == [[]]:
                del kv_pairs[-1]
            # value
            kv_pairs[-1].append([])  
        else:
            kv_pairs[-1][-1].append(char)

    kv_dict = {}

    for kv_pair in kv_pairs:
        # empty line
        if len(kv_pair) == 0:  
            continue
        # default to True
        elif len(kv_pair) == 1:  
            key, value = ''.join(kv_pair[0]), 'T'
        # Smush anything else with kv-splitter '=' between them
        else:  
            key, value = ''.join(kv_pair[0]), '='.join(
                ''.join(x) for x in kv_pair[1:])

        if key.lower() not in UNPROCESSED_KEYS:
            # Try to convert to (arrays of) floats, ints
            split_value = re.findall(r'[^\s,]+', value)
            try:
                try:
                    numvalue = np.array(split_value, dtype=int)
                except (ValueError, OverflowError):
                    # don't catch errors here so it falls through to bool
                    numvalue = np.array(split_value, dtype=float)
                if len(numvalue) == 1:
                    numvalue = numvalue[0]  # Only one number
                elif len(numvalue) == 9:
                    # special case: 3x3 matrix, fortran ordering
                    numvalue = np.array(numvalue).reshape((3, 3), order='F')
                value = numvalue
            except (ValueError, OverflowError):
                pass  # value is unchanged

            # Parse boolean values: 'T' -> True, 'F' -> False,
            #                       'T T F' -> [True, True, False]
            if isinstance(value, str):
                str_to_bool = {'T': True, 'F': False}

                try:
                    boolvalue = [str_to_bool[vpart] for vpart in
                                 re.findall(r'[^\s,]+', value)]
                    if len(boolvalue) == 1:
                        value = boolvalue[0]
                    else:
                        value = boolvalue
                except KeyError:
                    # value is unchanged
                    pass  

        kv_dict[key] = value

    return kv_dict


def parse_properties(prop_str):
    """
    Parse extended XYZ properties format string

    Format is "[NAME:TYPE:NCOLS]...]", e.g. "species:S:1:pos:R:3".
    NAME is the name of the property.
    TYPE is one of R, I, S, L for real, integer, string and logical.
    NCOLS is number of columns for that property.
    """

    properties = {}
    properties_list = []
    dtypes = []
    converters = []

    fields = prop_str.split(':')

    def parse_bool(x):
        """
        Parse bool to string
        """
        return {'T': True, 'F': False,
                'True': True, 'False': False}.get(x)

    fmt_map = {'R': ('d', float),
               'I': ('i', int),
               'S': (object, str),
               'L': ('bool', parse_bool)}

    for name, ptype, cols in zip(fields[::3],
                                 fields[1::3],
                                 [int(x) for x in fields[2::3]]):
        if ptype not in ('R', 'I', 'S', 'L'):
            raise ValueError('Unknown property type: ' + ptype)
        base_name = REV_PROPERTY_NAME_MAP.get(name, name)

        dtype, converter = fmt_map[ptype]
        if cols == 1:
            dtypes.append((name, dtype))
            converters.append(converter)
        else:
            for c in range(cols):
                dtypes.append((name + str(c), dtype))
                converters.append(converter)

        properties[name] = (base_name, cols)
        properties_list.append(name)

    dtype = np.dtype(dtypes)
    return properties, properties_list, dtype, converters


def _read_xyz_frame(lines, natoms, properties_parser=key_val_str_to_dict, nvec=0):
    # comment line
    data = {}
    line = next(lines).strip()
    if nvec > 0:
        info = {'comment': line}
    else:
        info = properties_parser(line) if line else {}

    if 'pbc' in info:
        data['pbc'] = info['pbc']
        del info['pbc']
    elif 'Lattice' in info:
        # default pbc for extxyz file containing Lattice
        # is True in all directions
        data['pbc'] = [True, True, True]
    elif nvec > 0:
        # cell information given as pseudo-Atoms
        data['pbc'] = [False, False, False]

    if 'Lattice' in info:
        # NB: ASE cell is transpose of extended XYZ lattice
        data['Lattice'] = info['Lattice']
        del info['Lattice']
    elif nvec > 0:
        # cell information given as pseudo-Atoms
        data['lattice'] = np.zeros((3, 3))

    if 'Properties' not in info:
        # Default set of properties is atomic symbols and positions only
        info['Properties'] = 'species:S:1:pos:R:3'
    properties, names, dtype, convs = parse_properties(info['Properties'])
    del info['Properties']

    atomic_data = []
    for ln in range(natoms):
        try:
            line = next(lines)
        except StopIteration:
            raise StopIteration('xyz scraper: Frame has {} atoms, expected {}'.format(len(atomic_data), natoms))
        vals = line.split()
        row = tuple([conv(val) for conv, val in zip(convs, vals)])
        atomic_data.append(row)

    try:
        atomic_data = np.array(atomic_data, dtype)
    except TypeError:
        raise TypeError('Badly formatted data or end of file reached before end of frame')

    arrays = {}
    for name in names:
        base_name, cols = properties[name]
        if cols == 1:
            value = atomic_data[name]
        else:
            value = np.vstack([atomic_data[name + str(c)]
                               for c in range(cols)]).T
        arrays[base_name] = value

    symbols = None
    if 'symbols' in arrays:
        data['AtomTypes'] = [s.capitalize() for s in arrays['symbols']]
        del arrays['symbols']

    return data, arrays, info


class XYZ(Scraper):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self.conversions = copy(self.default_conversions)
        self.all_data = []
        self.style_info = {}

    def scrape_groups(self):
        # Reset as empty dict in case running scrape twice.
        self.files = {}
        self.configs = {}

        pt = self.pt
        config = self.config
        if self.config.sections["SCRAPER"].save_group_scrape != "None":
            save_file = self.config.sections["PATH"].relative_directory + '/' + self.config.sections["SCRAPER"].save_group_scrape
            if pt.get_rank() == 0:
                with open(save_file, 'w') as fp:
                    fp.write('')
        else:
            save_file = None
        if self.config.sections["SCRAPER"].read_group_scrape != "None":
            if self.config.sections["SCRAPER"].save_group_scrape != "None":
                raise RuntimeError("Do not set both reading and writing of group_scrape")
            read_file = self.config.sections["PATH"].relative_directory + '/' + self.config.sections["SCRAPER"].read_group_scrape
        else:
            read_file = None

        group_dict = {k: self.config.sections["GROUPS"].group_types[i]
                      for i, k in enumerate(self.config.sections["GROUPS"].group_sections)}
        self.group_table = self.config.sections["GROUPS"].group_table
        size_type = None
        testing_size_type = None
        folder_files = listdir(self.config.sections["PATH"].datapath)

        for key in self.group_table:
            bc_bool = False
            training_size = None
            if 'size' in self.group_table[key]:
                training_size = self.group_table[key]['size']
                bc_bool = True
                size_type = group_dict['size']
            if 'training_size' in self.group_table[key]:
                if training_size is not None:
                    raise ValueError("Do not set both size and training size")
                training_size = self.group_table[key]['training_size']
                size_type = group_dict['training_size']
            if 'testing_size' in self.group_table[key]:
                testing_size = self.group_table[key]['testing_size']
                testing_size_type = group_dict['testing_size']
            else:
                testing_size = 0
            if training_size is None:
                raise ValueError("Please set training size for {}".format(key))

            file_base = path.join(self.config.sections["PATH"].datapath, key)

            if file_base.split('/')[-1] + ".extxyz" in folder_files:
                file_name = file_base + ".extxyz"
            elif file_base.split('/')[-1] + ".xyz" in folder_files:
                file_name = file_base + ".xyz"
            else:
                raise FileNotFoundError("{}.xyz not found in {}".format(file_base, self.config.sections["PATH"].datapath))

            if file_base + '.xyz' not in self.files or file_base + '.extxyz':
                self.files[file_base] = []
                self.configs[file_base] = []
            else:
                raise FileExistsError("{} was already found".format(file_base))

            self.files[file_base].append(file_name)

            if self.config.sections["SCRAPER"].read_group_scrape != "None":
                with open(read_file, 'r') as fp:
                    for line in fp:
                        split_line = line.split()
                        if split_line[0] == file_base:
                            for element in split_line[1:]:
                                self.configs[file_base].append(int(element))
            else:
                try:
                    with open(self.files[file_base][0], 'r') as fp:
                        lines = fp.readlines()
                        fp.seek(0)
                        self.configs[file_base].append(0)
                        count = lines[0]
                        line_number = 0
                        while True:
                            update = int(count) + 2
                            line_number += update
                            for i in range(update):
                                fp.readline()
                            self.configs[file_base].append(fp.tell())
                            count = lines[line_number]
                except IndexError:
                    self.configs[file_base].pop(-1)
            if config.sections["SCRAPER"].save_group_scrape != "None":
                if pt.get_rank() == 0:
                    with open(save_file, 'a') as fp:
                        fp.write("{}".format(file_base))
                        for item in self.configs[file_base]:
                            fp.write(" {}".format(item))
                        fp.write("\n")

            if config.sections["GROUPS"].random_sampling:
                shuffle(self.configs[file_base], random)
            nconfigs = len(self.configs[file_base])
            if training_size < 1 or (training_size == 1 and size_type == float):
                if training_size == 1:
                    training_size = abs(training_size) * nconfigs
                elif training_size == 0:
                    pass
                else:
                    training_size = max(1, int(abs(training_size) * nconfigs + 0.5))
                if bc_bool and testing_size == 0:
                    testing_size = nconfigs - training_size
            if testing_size != 0 and (testing_size < 1 or (testing_size == 1 and testing_size_type == float)):
                testing_size = max(1, int(abs(testing_size) * nconfigs + 0.5))
            training_size = self._float_to_int(training_size)
            testing_size = self._float_to_int(testing_size)
            if nconfigs - testing_size - training_size < 0:
                raise ValueError("training size: {} + testing size: {} is greater than files in folder: {}".format(
                    training_size, testing_size, nconfigs))
            pt.single_print(key, ": Detected ", nconfigs, " fitting on ", training_size, " testing on ", testing_size)
            if self.tests is None:
                self.tests = {}
            self.tests[file_base] = []
            for i in range(nconfigs - training_size - testing_size):
                self.configs[file_base].pop()
            for i in range(testing_size):
                self.tests[file_base].append(self.configs[file_base].pop())

            self.group_table[key]['training_size'] = training_size
            self.group_table[key]['testing_size'] = testing_size
            # self.files[folder] = natsorted(self.files[folder])

    def scrape_configs(self):
        """
        Scrape configs defined in the iterables of files and configs.

        Returns a list of data dictionaries containing structural info.
        """
        self.all_data = [] # Reset to empty list in case running scraper twice.

        pt = self.pt
        config = self.config
        for folder_num, folder in enumerate(self.files):
            filename = self.files[folder][0]
            with open(filename) as file:
                for i, configuration in enumerate(self.configs):
                    if configuration[1] != folder:
                        continue
                    starting_line = configuration[0]
                    file.seek(starting_line)
                    try:
                        num_atoms = int(file.readline())
                    except ValueError:
                        file.seek(starting_line)
                        raise ValueError("bad frame: error at {}".format(file.readline()))
                    file.seek(starting_line)
                    file.readline()

                    data, arrays, info = _read_xyz_frame(file, num_atoms)

                    # TODO: Implement Styles in xyz_scraper for units to be defined in comment line
                    for key in config.sections["SCRAPER"].properties:
                        if key.lower() in arrays:
                            data[key] = arrays[key.lower()]
                            del arrays[key.lower()]
                        if key in arrays:
                            data[key] = arrays[key]
                            del arrays[key]
                        if key.lower() in info:
                            data[key] = info[key.lower()]
                            del info[key.lower()]
                        if key in info:
                            data[key] = info[key]
                            del info[key]

                    if 'positions' in arrays:
                        data['Positions'] = arrays['positions']
                        del arrays['positions']

                    self.data = data

                    self.data['NumAtoms'] = num_atoms

                    self.data['Group'] = ".".join(filename.split("/")[-1].split(".")[:-1])
                    self.data['File'] = filename.split("/")[-1]

                    pt.shared_arrays["number_of_atoms"].sliced_array[i] = self.data['NumAtoms']

                    self.data["QMLattice"] = self.data["Lattice"] * self.conversions["Lattice"]

                    # Populate with LAMMPS-normalized lattice
                    del self.data["Lattice"]  

                    # TODO Check whether "Label" container useful to keep around
                    if "Label" in self.data:
                        del self.data["Label"]  

                    # possibly due to JSON, some configurations have integer energy values.
                    if not isinstance(self.data["Energy"], float):
                        # pt.print(f"Warning: Configuration {all_index}
                        # ({group_name}/{fname_end}) gives energy as an integer")
                        self.data["Energy"] = float(self.data["Energy"])

                    if hasattr(config.sections["ESHIFT"], 'eshift'):
                        try:
                            for atom in self.data["AtomTypes"]:
                                self.data["Energy"] += config.sections["ESHIFT"].eshift[atom]
                        except KeyError:
                            raise KeyError("{} not found in atom types".format(atom))

                    self.data["test_bool"] = self.test_bool[i]

                    self.data["Energy"] *= self.conversions["Energy"]

                    self._rotate_coords()
                    self._translate_coords()

                    self._weighting(num_atoms)

                    self.all_data.append(self.data)

        temp_list = []
        temp_list[:] = [d for d in self.all_data if d.get('test_bool') == 1]
        self.all_data[:] = [d for d in self.all_data if d.get('test_bool') == 0]
        self.all_data += temp_list

        return self.all_data
