from fitsnap3.io.sections.sections import Section
from os import sysconf
from subprocess import check_output


class Memory(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self._check_memory()
        self.memory = self.get_value("MEMORY", "memory", "{}".format(self.mem_bytes), "int")
        self.override = self.get_value("MEMORY", "override", "False", interpreter="bool")
        self.delete()

    def _check_memory(self):
        try:
            self.mem_bytes = sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES')
        except ValueError:
            self.mem_bytes = int(check_output(['sysctl', '-n', 'hw.memsize']).strip())
