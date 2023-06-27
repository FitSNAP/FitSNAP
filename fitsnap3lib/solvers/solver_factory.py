from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.solvers.ard import ARD
from fitsnap3lib.solvers.jax import JAX
from fitsnap3lib.solvers.lasso import LASSO
from fitsnap3lib.solvers.ridge import RIDGE
from fitsnap3lib.solvers.pytorch import PYTORCH
from fitsnap3lib.solvers.scalapack import ScaLAPACK
from fitsnap3lib.solvers.svd import SVD
from fitsnap3lib.solvers.tensorflowsvd import TensorflowSVD
from fitsnap3lib.solvers.anl import ANL
from fitsnap3lib.solvers.merr import MERR
from fitsnap3lib.solvers.network import NETWORK

#pt = ParallelTools()


def solver(solver_name, pt, cfg):
    """Solver Factory"""
    instance = search(solver_name)
    instance.__init__(solver_name, pt, cfg)
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
