Run FitSNAP
===========

There are two ways to run FitSNAP: (1) the python executable or (2) as a python library. The 
executable version is reserved for the most basic operation of fitting on a dataset, where data
is scraped from a directory, configurations are input to LAMMPS to calculate descriptors, and the 
machine learning problem is solved. The python library provides more flexibility and allows one to
modify the steps in that fitting process.

Executable
----------

Before using the executable, you must set the path to your FitSNAP directoy in your 
:code:`PYTHONPATH` environment variable, e.g.

.. code-block:: console

    export PYTHONPATH="path/to/FitSNAP:$PYTHONPATH"

which is conveniently placed in a :code:`~/.bashrc` or :code:`~/.bash_profile` file. Then fits
can be performed with the executable by doing

.. code-block:: console

   python -m fitsnap3 input.in

where :code:`input.in` is a FitSNAP input script, which is explained in the next section.

Input Scripts
-------------

Input scripts contain settings that tell FitSNAP how to perform a fit. Our input scripts take the
form of configuration files with a format explained by 
`Python's native ConfigParser class <configparser_>`_. These configuration files are composed of 
sections, each of which contains keys with values, e.g. like::

    [SECTION1]
    key1 = value1
    key2 = value2

    [SECTION2]
    key3 = value3
    key4 = value4
    key5 = value5

.. _configparser: https://docs.python.org/3/library/configparser.html

In FitSNAP, each section declares a setting for a certain aspect of the machine learning problem.
For example we have a :code:`BISPECTRUM` section whose keys determine settings for the bispectrum 
descriptors that describe interatomic geometry, a :code:`CALCULATOR` section whose keys determine
which LAMMPS computes to use for calculating the descriptors, a :code:`SOLVER` section whose keys
determine which numerical solver to use for performing the fit, and so forth.

There are many examples on the GitHub repo, for example the linear SNAP tantalum example has the 
following input script::

    [BISPECTRUM]
    numTypes = 1
    twojmax = 6
    rcutfac = 4.67637
    rfac0 = 0.99363
    rmin0 = 0.0
    wj = 1.0
    radelem = 0.5
    type = Ta
    wselfallflag = 0
    chemflag = 0
    bzeroflag = 0
    quadraticflag = 0

    [CALCULATOR]
    calculator = LAMMPSSNAP
    energy = 1
    force = 1
    stress = 1

    [ESHIFT]
    Ta = 0.0

    [SOLVER]
    solver = SVD
    compute_testerrs = 1
    detailed_errors = 1

    [SCRAPER]
    scraper = JSON

    [PATH]
    dataPath = JSON

    [OUTFILE]
    metrics = Ta_metrics.md
    potential = Ta_pot

    [REFERENCE]
    units = metal
    atom_style = atomic
    pair_style = hybrid/overlay zero 10.0 zbl 4.0 4.8
    pair_coeff1 = * * zero
    pair_coeff2 = * * zbl 73 73

    [GROUPS]
    # name size eweight fweight vweight
    group_sections = name training_size testing_size eweight fweight vweight
    group_types = str float float float float float
    smartweights = 0
    random_sampling = 0
    Displaced_A15 =  1.0    0.0       100             1               1.00E-08
    Displaced_BCC =  1.0    0.0       100             1               1.00E-08
    Displaced_FCC =  1.0    0.0       100             1               1.00E-08
    Elastic_BCC   =  1.0    0.0     1.00E-08        1.00E-08        0.0001
    Elastic_FCC   =  1.0    0.0     1.00E-09        1.00E-09        1.00E-09
    GSF_110       =  1.0    0.0      100             1               1.00E-08
    GSF_112       =  1.0    0.0      100             1               1.00E-08
    Liquid        =  1.0    0.0       4.67E+02        1               1.00E-08
    Surface       =  1.0    0.0       100             1               1.00E-08
    Volume_A15    =  1.0    0.0      1.00E+00        1.00E-09        1.00E-09
    Volume_BCC    =  1.0    0.0      1.00E+00        1.00E-09        1.00E-09
    Volume_FCC    =  1.0    0.0      1.00E+00        1.00E-09        1.00E-09

    [EXTRAS]
    dump_descriptors = 1
    dump_truth = 1
    dump_weights = 1
    dump_dataframe = 1

    [MEMORY]
    override = 0

We explain the sections and their keys in more detail below.

[BISPECTRUM]
^^^^^^^^^^^^

[CALCULATOR]
^^^^^^^^^^^^

[ESHIFT]
^^^^^^^^

[SOLVER]
^^^^^^^^

Library
-------

FitSNAP may also be run through the library interface. More documentation coming soon, but for now
the `GitHub repo examples <libexamples_>`_ may help set up scripts for your needs.

.. _libexamples: https://github.com/FitSNAP/FitSNAP/tree/master/examples/library


