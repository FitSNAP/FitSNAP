from fitsnap3.io.sections.sections import Section
from distutils.util import strtobool
from os import getcwd, path


class Path(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        paths = getcwd().split('/')+args.infile.split('/')[:-1]
        working_directory = ''
        for directory in paths[:-1]:
            working_directory += directory + '/'
        working_directory += paths[-1]
        self.datapath = path.join(working_directory, self._config.get("PATH", "dataPath", fallback="JSON"))
        self.group_file = path.join(working_directory, self._config.get("PATH", "groupfile", fallback="grouplist.in"))
        self.smartweights = strtobool(self._config.get("PATH", "smartweights", fallback="0"))
        self.delete()
