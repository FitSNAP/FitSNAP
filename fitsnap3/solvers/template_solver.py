from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import pt
"""Methods you may or must override in new solvers"""


class Template(Solver):

    def __init__(self, name):
        super().__init__(name)

    # Solver must override perform_fit method
    @pt.sub_rank_zero
    def perform_fit(self):
        """"""
        pass
