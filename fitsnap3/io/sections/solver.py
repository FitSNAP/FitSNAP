from .sections import Section
from ...parallel_tools import pt

class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['solver','normalweight','normratio','compute_testerrs','detailed_errors', \
                       'nsam', 'cov_nugget', 'mcmc_num', 'mcmc_gamma']
        for value_name in config['SOLVER']:
            if value_name in allowedkeys: continue
            else:
                raise RuntimeError(">>> Found unmatched variable in SOLVER section of input: ", value_name)
                #pt.single_print(">>> Found unmatched variable in SOLVER section of input: ",value_name)

        self.solver = self.get_value("SOLVER", "solver", "SVD")
        self.true_multinode = 0
        if self.solver == "ScaLAPACK":
            self.true_multinode = 1
        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.nsam = self.get_value("SOLVER", "nsam", "1", "int")
        self.cov_nugget = self.get_value("SOLVER", "cov_nugget", "0.0", "float")
        self.mcmc_num = self.get_value("SOLVER", "mcmc_num", "10000", "int")
        self.mcmc_gamma = self.get_value("SOLVER", "mcmc_gamma", "0.01", "float")
        self.delete()
