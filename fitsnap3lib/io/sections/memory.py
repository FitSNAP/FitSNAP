from fitsnap3lib.io.sections.sections import Section
from os import sysconf
from subprocess import check_output
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Memory(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['memory', 'override']
        self._check_section()

        self._check_memory()
        self.memory = self.get_value("MEMORY", "memory", "{}".format(self.mem_bytes), "int")
        self.override = self.get_value("MEMORY", "override", "False", interpreter="bool")
        self.delete()

    def _check_memory(self):
        try:
            self.mem_bytes = sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')
        except ValueError:
            self.mem_bytes = int(check_output(['sysctl', '-n', 'hw.memsize']).strip())
