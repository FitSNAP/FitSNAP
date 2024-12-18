from fitsnap3lib.io.sections.sections import Section


class CMAES(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['population_size', 'sigma']
        self._check_section()

        self.population_size = self.get_value("CMAES", "population_size", "36", "int")
        self.sigma = self.get_value("CMAES", "sigma", "0.1", "float")

        self.delete()
