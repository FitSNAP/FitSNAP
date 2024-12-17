from fitsnap3lib.solvers.solver import Solver
import cma

"""Methods you may or must override in new solvers"""

class CMAES(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
