from fitsnap3.parallel_tools import pt
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
        num_types = config.sections["BISPECTRUM"].numtypes
        a_len = (self.number_of_files_per_node + 3 * self.number_of_atoms +
                 self.number_of_files_per_node * 6)
        a_width = config.sections["BISPECTRUM"].ncoeff * num_types
        if not config.sections["MODEL"].bzeroflag:
            pt.create_shared_array('a', a_len, a_width + num_types)
        else:
            pt.create_shared_array('a', a_len, a_width)
        pt.create_shared_array('b', a_len)
        pt.create_shared_array('w', a_len)
        output.screen("'a' takes up ", 100*pt.shared_arrays['a'].get_memory()/pt.get_ram(), "% of the total memory")
        if pt.shared_arrays['a'].get_memory()/pt.get_ram() > 0.5:
            raise MemoryError("The 'a' matrix is larger than 50% of your RAM. \n Aborting...!")
        pt.slice_array('a', num_types=num_types)

    def process_configs(self, data, i):
        pass
