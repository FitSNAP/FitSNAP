from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
"""Methods you may or must override in new solvers"""


pt = ParallelTools()


class Template(Solver):

    def __init__(self, name):
        super().__init__(name)

    # Solver must override perform_fit method
    @pt.sub_rank_zero
    def perform_fit(self):
        """"""
        pass
