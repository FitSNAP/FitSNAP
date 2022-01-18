from ..sections import Section


class Ard(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['alphabig', 'alphasmall', 'lambdabig', 'lambdasmall', 'threshold_lambda']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alphabig = self.get_value("SOLVER", "alphabig", "1.0E-12", "float")
        self.alphasmall = self.get_value("SOLVER", "alphasmall", "1.0E-14", "float")
        self.lambdabig = self.get_value("SOLVER", "lambdabig", "1.0E-6", "float")
        self.lambdasmall = self.get_value("SOLVER", "lambdasmall", "1.0E-6", "float")
        self.threshold_lambda = self.get_value("SOLVER", "threshold_lambda", "100000", "int")
        self.delete()
