from fitsnap3.solvers.solver import Solver, pt
from fitsnap3.solvers.ard import ARD
from fitsnap3.solvers.jax import JAX
from fitsnap3.solvers.lasso import LASSO
from fitsnap3.solvers.pytorch import PYTORCH
from fitsnap3.solvers.scalapack import ScaLAPACK
from fitsnap3.solvers.svd import SVD
from fitsnap3.solvers.tensorflowsvd import TensorflowSVD


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
