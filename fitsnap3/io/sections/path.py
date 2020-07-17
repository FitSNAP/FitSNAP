from fitsnap3.io.sections.sections import Section
from os import path


class Path(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.relative_directory = self._get_relative_directory(self)
        self.datapath = path.join(self.relative_directory, self.get_value("PATH", "dataPath", "JSON"))
        self.group_file = path.join(self.relative_directory, self.get_value("PATH", "groupFile", "grouplist.in"))
        self.delete()
