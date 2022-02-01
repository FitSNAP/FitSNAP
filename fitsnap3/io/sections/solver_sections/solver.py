from ..sections import Section


class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['solver', 'normalweight', 'normratio', 'compute_testerrs', 'detailed_errors','alpha',
                            'max_iter','alphabig','alphasmall,','lambdabig','lambdasmall','threshold_lambda']
        self._check_section()

        self.solver = self.get_value("SOLVER", "solver", "SVD")

        self.true_multinode = 0
        if self.solver == "ScaLAPACK":
            self.true_multinode = 1

        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.alpha = self.get_value("SOLVER", "alpha", "1E-6", "float")
        self.alphabig = self.get_value("SOLVER", "alphabig", "1E-12", "float")
        self.alphasmall = self.get_value("SOLVER", "alphasmall", "1E-12", "float")
        self.lambdabig = self.get_value("SOLVER", "lambdabig", "1E-6", "float")
        self.lambdasmall = self.get_value("SOLVER", "lambdasmall", "1E-6", "float")
        self.max_iter = self.get_value("SOLVER", "max_iter", "10000", "int")
        self.threshold_lambda = self.get_value("SOLVER", "threshold_lambda", "210000", "int")
        self.delete()
