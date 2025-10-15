from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.solvers.ard import ARD
from fitsnap3lib.solvers.jax import JAX
from fitsnap3lib.solvers.lasso import LASSO
from fitsnap3lib.solvers.ridge import RIDGE
from fitsnap3lib.solvers.pytorch import PYTORCH
from fitsnap3lib.solvers.svd import SVD
from fitsnap3lib.solvers.tensorflowsvd import TensorflowSVD
from fitsnap3lib.solvers.anl import ANL
from fitsnap3lib.solvers.merr import MERR
from fitsnap3lib.solvers.network import NETWORK
from fitsnap3lib.solvers.slate import SLATE

#pt = ParallelTools()


def solver(solver_name, pt, cfg):
    """Solver Factory"""
    instance = search(solver_name)
    instance.__init__(solver_name, pt, cfg)
    return instance

def search(solver_name):
    instance = None

    def find_subclass_recursive(base_class, target_name):
        """Recursively search through all subclass levels"""
        # Check the current class
        if base_class.__name__.lower() == target_name.lower():
            return base_class
        
        # Check all direct subclasses
        for subclass in base_class.__subclasses__():
            result = find_subclass_recursive(subclass, target_name)
            if result is not None:
                return result
        
        return None
    
    # Find the target class recursively
    target_class = find_subclass_recursive(Solver, solver_name)
    
    if target_class is not None:
        instance = Solver.__new__(target_class)
    
    if instance is None:
        raise IndexError("{} was not found in fitsnap solvers".format(solver_name))
    else:
        return instance
