Solver
======

FitSNAP uses a :code:`Solver` class which is a parent of all the different types of solvers, e.g. SVD and
ARD for linear regression, `PYTORCH` and `JAX` for neural networks, etc.

.. automodule:: fitsnap3lib.solvers.solver
    :members:

Specific solvers are inherit from the base :code:`Solver` class:

PYTORCH
-------

This class inherits from the `Solver` class, since it is a particular solver option. 

.. automodule:: fitsnap3lib.solvers.pytorch
    :members: