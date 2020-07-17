from fitsnap3.io.sections.sections import Section


class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.solver = self.get_value("SOLVER", "solver", "SVD")
        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.multinode_testing = self.get_value("SOLVER", "multinode_testing", "0", "bool")
        self.apply_transpose = self.get_value("SOLVER", "apply_transpose", "0", "bool")
        self.delete()
