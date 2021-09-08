from .sections import Section
from os import sysconf
from subprocess import check_output
from ...parallel_tools import ParallelTools


pt = ParallelTools()


class Memory(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['memory','override']
        for value_name in config['MEMORY']:
            if value_name in allowedkeys: continue
            else:
                raise RuntimeError(">>> Found unmatched variable in MEMORY section of input: ", value_name)
                #pt.single_print(">>> Found unmatched variable in MEMORY section of input: ",value_name)

        self._check_memory()
        self.memory = self.get_value("MEMORY", "memory", "{}".format(self.mem_bytes), "int")
        self.override = self.get_value("MEMORY", "override", "False", interpreter="bool")
        self.delete()

    def _check_memory(self):
        try:
            self.mem_bytes = sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')
        except ValueError:
            self.mem_bytes = int(check_output(['sysctl', '-n', 'hw.memsize']).strip())
