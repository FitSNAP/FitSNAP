"""Groups Class"""
from fitsnap3lib.io.sections.sections import Section
from pandas import read_csv
from os import path
#from fitsnap3lib.parallel_tools import ParallelTools
#from fitsnap3lib.parallel_output import Output


#pt = ParallelTools()
#output = Output()


def _str_2_fun(some_list):
    for i, item in enumerate(some_list):
        if item == 'str':
            some_list[i] = str
        if item == 'bool':
            some_list[i] = bool
        if item == 'int':
            some_list[i] = int
        if item == 'float':
            some_list[i] = float


class Groups(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['group_sections', 'group_types', 'smartweights', 'random_sampling', 'random_seed', 'vasp_use_TOTEN','vasp_json_pathname','vasp_ignore_incomplete','vasp_ignore_jsons','vasp_unconverged_label','BOLTZ']

        # for value_name in config['GROUPS']:
        #     if value_name in allowedkeys: continue
        #      else: pt.single_print(">>> Found unmatched variable in GROUPS section of input: ",value_name)

        self.group_sections = self.get_value("GROUPS", "group_sections", "name size eweight fweight vweight").split()
        self.group_types = self.get_value("GROUPS", "group_types", "str float float float float").split()
        self.smartweights = self.get_value("GROUPS", "smartweights", "0", "bool")
        self.random_sampling = self.get_value("GROUPS", "random_sampling", "0", "bool")
        self.random_seed = self.get_value("GROUPS", "random_seed", "0", "float")
        self.vasp_use_TOTEN = self.get_value("GROUPS", "vasp_use_TOTEN", "False", "bool")
        self.vasp_json_pathname = self.get_value("GROUPS", "vasp_json_pathname", "vJSON", "str")
        self.vasp_ignore_incomplete = self.get_value("GROUPS", "vasp_ignore_incomplete", "1", "bool")
        self.vasp_ignore_jsons = self.get_value("GROUPS", "vasp_ignore_jsons", "0", "bool")
        self.vasp_unconverged_label = self.get_value("GROUPS", "vasp_unconverged_label", "UNCONVERGED", "str")
        self.boltz = self.get_value("BISPECTRUM", "BOLTZ", "0", "float")
        _str_2_fun(self.group_types)
        self.group_table = None
        if self.get_value("PATH", "groupFile", "None") != "None":
            self.read_group_file()
        else:
            self.read_group_config()

        # run init methods to populate class
        # delete config and args to rely only on child's members
        self.delete()

    def read_group_config(self):
        try:
            self.group_table = {k: v.split() for (k, v) in self.get_section("GROUPS")}
        except TypeError:
            raise FileNotFoundError("Group File not found, make sure Input File can be found")
        # Deletes any key:value from self.groups that shares a key name with the vars of this instance
        for k in vars(self):
            if k in self.group_table:
                self.group_table.pop(k)
        for k, v in self.group_table.items():
            expected_variables = len(self.group_sections[1:]) ## exclude duplicated 'name' column (same as group)
            found_variables = len(v) 

            if found_variables != expected_variables:
                ## There is probably a typo somewhere, let user know
                raise Exception('!!ERROR: Too many group variables found!!' 
                        '\n!!Check the input file section [GROUP] for extra variables, typos ' 
                        f'\n!!\tInput file: {self._args.infile}' 
                        f'\n!!\tGroup line: {k} = {v}' 
                        f'\n!!\tExpected {expected_variables} columns for settings, found {found_variables}'
                        '\n')
            self.group_table[k] = {self.group_sections[i+1]: self.group_types[i+1](item) for i, item in enumerate(v)}

    def read_group_file(self):
        group_types = {self.group_sections[i]: item for i, item in enumerate(self.group_types)}
        group_table = read_csv(path.join(Section.get_infile_directory(self), self.get_value("PATH", "groupFile", "grouplist.in")),
                               delim_whitespace=True,
                               comment='#',
                               skip_blank_lines=True,
                               names=self.group_sections,
                               index_col=False)

        # Remove blank lines ; skip_blank_lines doesn't seem to work.
        group_table = group_table.dropna()
        group_table.index = range(len(group_table.index))

        # Convert data types
        group_table = group_table.astype(dtype=group_types)
        self.group_table = {}
        for data_frame in group_table.itertuples():
            self.group_table[data_frame[1]] = {self.group_sections[i+1]: item for i, item in enumerate(data_frame[2:])}
