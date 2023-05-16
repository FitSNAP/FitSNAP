.. _Library:

Library
=======

The FitSNAP library provides a high level connection to FitSNAP methods in external Python scripts.
The library is designed based on instances of :code:`FitSnap` objects with some important points:

* Each :code:`FitSnap` instance possesses its own settings, such as hyperparameters.
* Each :code:`FitSnap` instance possesses its own optional MPI communicator over which appropriate 
  operations, such as calculating descriptors, are parallelized.
* All results of collating data, calculating descriptors, and fitting a potential are therefore 
  contained within a :code:`FitSnap` instance; this improves organization of fits and reduces 
  confusion about where a trained model came from.

To use the library we must first import :code:`FitSnap`::

    from fitsnap3lib.fitsnap import FitSnap

We will create an instance of :code:`FitSnap` with specific input settings.
First we need to define the settings used by :code:`FitSnap`. This can be a path to a traditional 
input script, or a dictionary containing sections and keywords. For example a :code:`settings` 
dictionary to perform a fit can be defined like::

    settings = \
    {
    "BISPECTRUM":
        {
        "numTypes": 1,
        "twojmax": 6,
        "rcutfac": 4.67637,
        "rfac0": 0.99363,
        "rmin0": 0.0,
        "wj": 1.0,
        "radelem": 0.5,
        "type": "Ta"
        },
    "CALCULATOR":
        {
        "calculator": "LAMMPSSNAP",
        "energy": 1,
        "force": 1,
        "stress": 1
        },
    "SOLVER":
        {
        "solver": "SVD"
        },
    "SCRAPER":
        {
        "scraper": "JSON" 
        },
    "PATH":
        {
        "dataPath": "/Users/adrohsk/FitSNAP/examples/Ta_Linear_JCP2014/JSON"
        },
    "REFERENCE":
        {
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "hybrid/overlay zero 6.0 zbl 4.0 4.8",
        "pair_coeff1": "* * zero",
        "pair_coeff2": "* * zbl 73 73"
        },
    "GROUPS":
        {
        "group_sections": "name training_size testing_size eweight fweight vweight",
        "group_types": "str float float float float float",
        "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
        "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
        }
    }

Create an :code:`FitSnap` instance using these settings with::

    # The --overwrite command line arg lets us overwrite possible output files.
    instance = FitSnap(settings, arglist=["--overwrite"])

Then use the *high level* functions for (1) scraping data, (2) calculating descriptors, and (3) 
performing a fit::

    # Scrape fitting data.
    instance.scrape_configs()
    # Calculate descriptors.
    instance.process_configs()
    # Fit the model.
    instance.perform_fit()
    # Observe the errors.
    print(instance.solver.errors)

Each :code:`FitSnap` instance contains its own settings for defining an entire machine learning fit 
from start to finish. 
This can include training data and hyperparameters all the way to the final fitting coefficients or 
model and error metrics. 
This design is similar to scikit-learn, where users make instances out of model classes like 
:code:`instance = Ridge(alpha)` and call class methods such as :code:`instance.fit(A, b)`. 
With :code:`FitSnap`, however, we have many more settings and hyperparameters. 
It therefore improves organization to contain all these attributes in a single :code:`FitSnap` 
instance to reduce confusion about where a fit came from.
Most methods such as calculating descriptors and performing fits are methods of a particular 
instance, and the actions of these methods depend on the state or settings of that instance.
These methods and the rest of the API are detailed below.

FitSnap
-------

.. automodule:: fitsnap3lib.fitsnap
    :members:

:code:`FitSnap` contains instances of two helper classes that help with MPI communication and 
settings: ParallelTools and Config are explained below.

Parallel Tools
^^^^^^^^^^^^^^

.. automodule:: fitsnap3lib.parallel_tools
    :members:

Config
^^^^^^

.. automodule:: fitsnap3lib.io.input
    :members:

Scraper
-------

.. automodule:: fitsnap3lib.scrapers.scrape
    :members:

Solver
------

FitSNAP uses a :code:`Solver` class which is a parent of all the different types of solvers, e.g. SVD and
ARD for linear regression, `PYTORCH` and `JAX` for neural networks, etc.

.. automodule:: fitsnap3lib.solvers.solver
    :members:

Specific solvers are inherit from the base :code:`Solver` class:

PYTORCH
^^^^^^^

This class inherits from the `Solver` class, since it is a particular solver option. 

.. automodule:: fitsnap3lib.solvers.pytorch
    :members:

lib/
----

The `fitsnap3lib/lib` directory contains external code used by FitSNAP, sort of like helper classes
and functions. 

Atom-centered Networks
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fitsnap3lib.lib.neural_networks.pytorch
    :members:

Pairwise Networks
^^^^^^^^^^^^^^^^^

.. automodule:: fitsnap3lib.lib.neural_networks.pairwise
    :members:
