from fitsnap3lib.io.sections.sections import Section


class Adaptive_Ridge(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha','local_solver','max_coeff','max_vmr']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alpha = self.get_value("ADAPTIVE_RIDGE", "alpha", "1.0E-8", "float")
        self.local_solver = self.get_value("ADAPTIVE_RIDGE", "local_solver", "1", "bool")
        self.maxcoeff = self.get_value("ADAPTIVE_RIDGE", "max_coeff", "5.0", "float")
        self.maxvmr = self.get_value("ADAPTIVE_RIDGE", "max_vmr", "2.0", "float")

        self.delete()
