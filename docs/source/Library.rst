.. _Library:

Library
=======

Overview
--------

The FitSNAP library provides a high level connection to FitSNAP methods in external Python scripts.

Before using the FitSNAP library, we must have the FitSNAP directory in our Python path, e.g. put 
the following in your `~/.bashrc` or `~/.bash_profile`:

.. code-block:: console
    
    export PYTHONPATH="/path/to/FitSNAP:$PYTHONPATH"

A sequence of models must be loaded to first use the FitSNAP library, specifically the
`ParallelTools` and `Config` modules. This is seen in the `examples/library` directory, where we do
the following imports before creating an instance `snap` of the FitSNAP library:

>>> from fitsnap3lib.parallel_tools import ParallelTools
>>> pt = ParallelTools()
>>> from fitsnap3lib.io.input import Config
>>> config = Config(arguments_lst = ["/path/to/FitSNAP/input/script", "--overwrite"])
>>> from fitsnap3lib.fitsnap import FitSnap
>>> snap = FitSnap()

We must load `ParallelTools` because it is the backbone of FitSNAP - storing all the data arrays
upon which fitting depends on. 

Parallel Tools
--------------

.. automodule:: fitsnap3lib.parallel_tools
    :members:


After a `ParallelTools` tools instance is created, we may import `Config` which stores the settings
of our fit.

Config
------

FitSNAP uses the `ConfigParser <https://docs.python.org/3/library/configparser.html>`_ module to parse input scripts and settings.

We use this to create a `Config` instance by supplying the path to our 
FitSNAP input script, which contains all the settings of the fit. 

.. automodule:: fitsnap3lib.io.input
    :members:

After creating an instance of the `Config` class, we are ready to create a FitSNAP object.

The `snap` object is now an instance of the `FitSNAP()` class, with the following

FitSNAP
-------

.. automodule:: fitsnap3lib.fitsnap
    :members:

Solver
------
FitSNAP uses a `Solver` class which is a parent of all the different types of solvers, e.g. SVD and
ARD for linear regression, `PYTORCH` and `JAX` for neural networks, etc.

.. automodule:: fitsnap3lib.solvers.solver
    :members:

PYTORCH
^^^^^^^

This class inherits from the `Solver` class, since it is a particular solver option. 

.. automodule:: fitsnap3lib.solvers.pytorch
    :members:

lib/
----

The `fitsnap3lib/lib` directory contains external code used by FitSNAP, sort of like helper classes
and functions. 

FitTorch
^^^^^^^^

.. automodule:: fitsnap3lib.lib.neural_networks.pytorch
    :members:
