from fitsnap3lib.io.sections.sections import Section

import cma

class CMAES(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['popsize', 'sigma']
        self._check_section()
        self.popsize = self.get_value("CMAES", "popsize", "10", "int")
        self.sigma = self.get_value("CMAES", "sigma", "0.1", "float")
        self.delete()

