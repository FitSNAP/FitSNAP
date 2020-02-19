from fitsnap3.io.sections.sections import Section
from distutils.util import strtobool


class Path(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.datapath = self._config.get("PATH", "dataPath", fallback="JSON")
        self.group_file = self._config.get("PATH", "groupfile", fallback="grouplist.in")
        self.smartweights = strtobool(self._config.get("PATH", "smartweights", fallback="0"))
        self.delete()
