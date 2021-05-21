from .sections import Section
from ...parallel_tools import pt

class Solver(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        allowedkeys = ['solver','normalweight','normratio','compute_testerrs','detailed_errors']
        for value_name in config['SOLVER']:
            if value_name in allowedkeys: continue
            else: pt.single_print(">>> Found unmatched variable in SOLVER section of input: ",value_name)

        self.solver = self.get_value("SOLVER", "solver", "SVD")
        self.normalweight = self.get_value("SOLVER", "normalweight", "-12", "float")
        self.normratio = self.get_value("SOLVER", "normratio", "0.5", "float")
        self.compute_testerrs = self.get_value("SOLVER", "compute_testerrs", "0", "bool")
        self.detailed_errors = self.get_value("SOLVER", "detailed_errors", "0", "bool")
        self.delete()
