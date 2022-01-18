from ..sections import Section


class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['solver', 'normalweight', 'normratio', 'compute_testerrs', 'detailed_errors']
        self._check_section()

        self.solver = self.get_value("SOLVER", "solver", "SVD")

        self.true_multinode = 0
        if self.solver == "ScaLAPACK":
            self.true_multinode = 1

        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.delete()
