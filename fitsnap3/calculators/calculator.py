from fitsnap3.parallel_tools import pt, double_size
from fitsnap3.io.input import config
from fitsnap3.io.output import output


class Calculator:

    def __init__(self, name):
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None

    def create_a(self):
        # TODO : Any extra config pulls should be done before this
        pt.sub_barrier()
        self.number_of_atoms = pt.shared_arrays["number_of_atoms"].array.sum()
        self.number_of_files_per_node = len(pt.shared_arrays["number_of_atoms"].array)

        elements = 0
        testing = pt.shared_arrays["configs_per_group"].testing
        if testing > 0:
            for i in range(testing):
                if config.sections["CALCULATOR"].energy:
                    elements += 1
                if config.sections["CALCULATOR"].force:
                    elements += 3 * pt.shared_arrays["number_of_atoms"].array[-testing+i]
                if config.sections["CALCULATOR"].stress:
                    elements += 6
        pt.shared_arrays["configs_per_group"].testing_elements = elements

        num_types = config.sections["BISPECTRUM"].numtypes

        a_len = 0
        if config.sections["CALCULATOR"].energy:
            a_len += self.number_of_files_per_node
        if config.sections["CALCULATOR"].force:
            a_len += 3 * self.number_of_atoms
        if config.sections["CALCULATOR"].stress:
            a_len += self.number_of_files_per_node * 6

        a_width = config.sections["BISPECTRUM"].ncoeff * num_types
        if not config.sections["CALCULATOR"].bzeroflag:
            a_width += num_types

        # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
        a_size = a_len * a_width * double_size
        # output.screen("'a' takes up ", 100 * a_size / pt.get_ram(), "% of the total memory")
        output.screen("'a' takes up ", 100 * a_size / config.sections["MEMORY"].memory,
                      "% of the total memory", config.sections["MEMORY"].memory*1e-9, "GB of RAM")
        if a_size / pt.get_ram() > 0.5 and not config.sections["MEMORY"].override:
            raise MemoryError("The 'a' matrix is larger than 50% of your RAM. \n Aborting...!")
        elif a_size / pt.get_ram() > 0.5 and config.sections["MEMORY"].override:
            output.screen("Warning: I hope you know what you are doing!")

        pt.create_shared_array('a', a_len, a_width)
        pt.create_shared_array('b', a_len)
        pt.create_shared_array('w', a_len)
        pt.slice_array('a', num_types=num_types)

    def process_configs(self, data, i):
        pass
