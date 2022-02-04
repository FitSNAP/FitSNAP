from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import ParallelTools

try:
    from fitsnap3.solvers.ard import ARD
except ImportError:
    pass

try:
    from fitsnap3.solvers.lasso import LASSO
except ImportError:
    pass

try:
    from fitsnap3.solvers.scalapack import ScaLAPACK
except ImportError:
    pass


pt = ParallelTools()
# pt.get_subclasses(__name__, __file__, Solver)


def solver(solver_name):
    """Solver Factory"""
    instance = search(solver_name)
    instance.__init__(solver_name)
    return instance


def search(solver_name):
    instance = None
    for cls in Solver.__subclasses__():
        if cls.__name__.lower() == solver_name.lower():
            instance = Solver.__new__(cls)

    if instance is None:
        raise IndexError("{} was not found in fitsnap solvers".format(solver_name))
    else:
        return instance
