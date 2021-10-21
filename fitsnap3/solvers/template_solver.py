from fitsnap3.solvers.solver import Solver
"""Methods you may or must override in new solvers"""


class Template(Solver):

    def __init__(self, name):
        super().__init__(name)

    # Solver must override perform_fit method
    def perform_fit(self):
        """"""
        pass
