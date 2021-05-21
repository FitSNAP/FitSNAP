from .sections import Section
from os import path
from ...parallel_tools import pt

class Path(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['dataPath','groupFile']
        for value_name in config['PATH']:
            if value_name in allowedkeys: continue
            else: pt.single_print(">>> Found unmatched variable in PATH section of input: ",value_name)

        self.relative_directory = self._get_relative_directory(self)
        self.datapath = path.join(self.relative_directory, self.get_value("PATH", "dataPath", "JSON"))
        self.group_file = path.join(self.relative_directory, self.get_value("PATH", "groupFile", "grouplist.in"))
        self.delete()
