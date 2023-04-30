from fitsnap3lib.io.sections.sections import Section
from os import path
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Path(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['dataPath', 'groupFile']
        self._check_section()

        # Set infile directory if we have an infile (as opposed to an indict):

        self.infile_directory = Section.get_infile_directory(self)
        print(f">>> sections.path infile dir: {self.infile_directory}")
        self.outfile_directory = Section.get_outfile_directory(self)
        print(f">>> sections.path outfile dir: {self.outfile_directory}")
        self.datapath = path.join(self.infile_directory, self.get_value("PATH", "dataPath", "JSON"))
        print(f">>> sections.path.datapath: {self.datapath}")
        self.group_file = path.join(self.infile_directory, self.get_value("PATH", "groupFile", "grouplist.in"))
        print(f">>> sections.path.group_file: {self.group_file}")
        self.delete()
