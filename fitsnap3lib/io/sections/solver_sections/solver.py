from fitsnap3lib.io.sections.sections import Section


class Solver(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['solver', 'normalweight', 'normratio', \
                            'compute_testerrs', 'detailed_errors', \
                            'nsam', 'cov_nugget', \
                            'mcmc_num', 'mcmc_gamma', \
                            'merr_mult', 'merr_method', "merr_cfs"]
        self._check_section()

        self.solver = self.get_value("SOLVER", "solver", "SVD")

        self.true_multinode = 0
        if self.solver == "ScaLAPACK":
            self.true_multinode = 1

        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.nsam = self.get_value("SOLVER", "nsam", "0", "int")
        self.cov_nugget = self.get_value("SOLVER", "cov_nugget", "0.0", "float")
        self.mcmc_num = self.get_value("SOLVER", "mcmc_num", "10000", "int")
        self.mcmc_gamma = self.get_value("SOLVER", "mcmc_gamma", "0.01", "float")
        self.merr_mult = self.get_value("SOLVER", "merr_mult", "0", "bool")
        self.merr_method = self.get_value("SOLVER", "merr_method", "abc", "str")
        self.merr_cfs = self.get_value("SOLVER", "merr_cfs", "all", "str")
        self.delete()
