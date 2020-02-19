from fitsnap3.io.sections.sections import Section
from distutils.util import strtobool


class Model(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.alloyflag = strtobool(self._config.get("MODEL", "alloyflag", fallback='0'))
        self.wselfallflag = strtobool(self._config.get("MODEL", "wselfallflag", fallback='0'))
        self.bzeroflag = strtobool(self._config.get("MODEL", "bzeroflag", fallback='0'))
        self.quadraticflag = strtobool(self._config.get("MODEL", "quadraticflag", fallback='0'))
        self.solver = self._config.get("MODEL", "solver", fallback='SVD')
        self.normalweight = float(self._config.get("MODEL", "normalweight", fallback='-12'))
        self.normratio = float(self._config.get("MODEL", "normratio", fallback='0.5'))
        self.compute_dbvb = strtobool(self._config.get("MODEL", "compute_dbvb", fallback='0'))
        self.compute_testerrs = strtobool(self._config.get("MODEL", "compute_testerrs", fallback='0'))
        self.delete()
