from fitsnap3lib.solvers.solver import Solver
"""Methods you may or must override in new solvers"""

class Template(Solver):

    def __init__(self, name):
        super().__init__(name)
