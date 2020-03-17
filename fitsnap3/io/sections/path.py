from fitsnap3.io.sections.sections import Section
from os import getcwd, path


class Path(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        paths = getcwd().split('/')+args.infile.split('/')[:-1]
        working_directory = ''
        for directory in paths[:-1]:
            working_directory += directory + '/'
        working_directory += paths[-1]
        self.datapath = path.join(working_directory, self.get_value("PATH", "dataPath", "JSON"))
        self.group_file = path.join(working_directory, self.get_value("PATH", "groupfile", "grouplist.in"))
        self.smartweights = self.get_value("PATH", "smartweights", "0", "bool")
        self.delete()
