Solver
======

FitSNAP uses a :code:`Solver` class which is a parent of all the different types of solvers, e.g. SVD and
ARD for linear regression, `PYTORCH` and `JAX` for neural networks, etc.

.. automodule:: fitsnap3lib.solvers.solver
    :members:

Specific solvers are inherited from the base :code:`Solver` class.

SVD
---

This class is for performing SVD fits on linear systems.

.. automodule:: fitsnap3lib.solvers.svd
    :members:

RIDGE
-----

This class is for performing ridge regression fits on linear systems.

.. automodule:: fitsnap3lib.solvers.ridge
    :members:

PYTORCH
-------

This class inherits from the `Solver` class, since it is a particular solver option. 

.. automodule:: fitsnap3lib.solvers.pytorch
    :members: