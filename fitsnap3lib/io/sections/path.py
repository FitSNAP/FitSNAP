from fitsnap3lib.io.sections.sections import Section
from os import path
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Path(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['dataPath', 'groupFile']
        self._check_section()

        self.infile_directory = Section.get_infile_directory(self)
        self.outfile_directory = Section.get_outfile_directory(self)
        self.datapath = path.join(self.infile_directory, self.get_value("PATH", "dataPath", "JSON"))
        self.group_file = path.join(self.infile_directory, self.get_value("PATH", "groupFile", "grouplist.in"))
        self.delete()
