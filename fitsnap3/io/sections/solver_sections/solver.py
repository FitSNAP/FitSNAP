from ..sections import Section


class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['solver', 'compute_testerrs', 'detailed_errors', 'normratio', 'normalweight']

        self.solver = self.get_value("SOLVER", "solver", "SVD")
        if self.solver == "LASSO":
            self.lasso()
        elif self.solver == "ARD":
            self.ard()
        
        self._check_section()

        self.true_multinode = 0
        if self.solver == "ScaLAPACK":
            self.true_multinode = 1

        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.delete()


    def lasso(self):
        self.allowedkeys.extend(['alpha', 'max_iter'])
        #self._check_if_used("SOLVER", "solver", "SVD")
        self.alpha = self.get_value("SOLVER", "alpha", "1.0E-8", "float")
        self.max_iter = self.get_value("SOLVER", "max_iter", "2000", "int")

    def ard(self):
        self.allowedkeys.extend(['alphabig', 'alphasmall', 'lambdabig', 'lambdasmall', 'threshold_lambda'])
        #self._check_if_used("SOLVER", "solver", "SVD")
        self.alphabig = self.get_value("SOLVER", "alphabig", "1.0E-12", "float")
        self.alphasmall = self.get_value("SOLVER", "alphasmall", "1.0E-14", "float")
        self.lambdabig = self.get_value("SOLVER", "lambdabig", "1.0E-6", "float")
        self.lambdasmall = self.get_value("SOLVER", "lambdasmall", "1.0E-6", "float")
        self.threshold_lambda = self.get_value("SOLVER", "threshold_lambda", "100000", "int")
