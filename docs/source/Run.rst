Run FitSNAP
===========

**If you want to get started immediately with interactive examples**, please see our 
`Colab Python notebook tutorial <tutorialnotebook_>`_

.. _tutorialnotebook: https://colab.research.google.com/github/FitSNAP/FitSNAP/blob/master/tutorial.ipynb

There are two ways to run FitSNAP: (1) the python executable or (2) as a python library. The 
executable version is reserved for the most basic operation of fitting on a dataset, where data
is scraped from a directory, configurations are input to LAMMPS to calculate descriptors, and the 
machine learning problem is solved. The python library provides more flexibility and allows one to
modify the steps in that fitting process.

.. _Run Executable:

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

This section contains settings for the SNAP bispectrum descriptors from `Thompson et. al. <snappaper_>`_

.. _snappaper: https://www.sciencedirect.com/science/article/pii/S0021999114008353

- :code:`numTypes` number of atom types in your set of configurations located in `the [PATH] section <Run.html#path>`__

- :code:`type` contains a list of element type symbols, one for each type. Make sure these are 
  ordered correctly, e.g. if you have a LAMMPS type 1 atom that is :code:`Ga`, and LAMMPS type 2 
  atoms are :code:`N`, list this as :code:`Ga N`.

The remaining keywords are thoroughly explained in the `LAMMPS docs on computing SNAP descriptors <lammpssnap_>`_ 
but we will give an overview here. **These are hyperparameters that *could* be optimized for your 
specific system, but this is not a requirement. You may also use the default values, or values used 
in our examples, which are often well behaved for other systems.**

- :code:`twojmax` determines the number of bispectrum coefficients for each element type. Give an 
  argument for each element type, e.g. for two element types we may use :code:`6 6` declaring 
  :code:`twojmax = 6` for each type. Higher :code:`twojmax` increases the number of bispectrum 
  components for each atom, thus potentially giving more accuracy at an increased cost. We recommend 
  using a :code:`twojmax` of 4, 6, or 8. This corresponds to 14, 30, and 55 bispectrum components, 
  respectively. Default value is 6. 

- :code:`rcutfac` is a cutoff radius parameter. One value is used for all element types. We recommend 
  a cutoff between 4 and 5 Angstroms for most systems. Default value is 4.67 Angstroms. 

- :code:`rfac0` is a parameter used in distance to angle conversion, between 0 and 1. Default value 
  is 0.99363.

- :code:`rmin0` another parameter used in distance to angle conversion, between 0 and 1. Default value 
  is 0.

- :code:`wj` list of neighbor weights. Give one argument for each element types, e.g. for two element 
  types we may use :code:`1.0 0.5` declaring a weight of 1.0 for neighbors of type 1, and 0.5 for 
  neighbors of type 2. We recommend taking values from the existing multi-element examples.

- :code:`radelem` list of cutoff radii, one for each element type. These values get multiplied by 
  :code:`rcutfac` to determine the effective cutoff of a particular type.

- :code:`wselfallflag` is 0 or 1, determining whether self-contribution is for elements of a central 
  atom or for all elements, respectively.

- :code:`chemflag` is 0 or 1, determining whether to use explicit multi-element SNAP descriptors as 
  explained in `Cusentino et. al. <chemsnappaper_>`_, and used in the InP example. This 
  increases the number of SNAP descriptors to resolve multi-element environment descriptions, and 
  therefore comes at an increase in cost but higher accuracy. This option is not required 
  for multi-element systems; the default value is 0.

- :code:`bzeroflag` is 0 or 1, determining whether or not B0, the bispectrum components of an atom 
  with no neighbors, are subtracted from the calculated bispectrum components.

- :code:`quadraticflag` is 0 or 1, determining whether or not to use quadratic descriptors in a 
  linear model, as done by `Wood and Thompson <quadsnappaper_>`_, and illusrated in the 
  :code:`Ta_Quadratic` example.

The following keywords are necessary for extracting per-atom descriptors and individual derivatives 
of bispectrum components with respect to neighbors, required for neural network potentials. See more 
info in `PyTorch Models <Pytorch.html>`__

- :code:`bikflag` is 0 or 1, determining whether to compute per-atom bispectrum descriptors instead 
  of sums of components for each atom. We do the latter for linear fitting because of the nature of 
  the linear problem, which saves memory, but per-atom descriptors are required for neural networks. 

- :code:`dgradflag` is 0 or 1, determining whether to compute individual derivatives of descriptors 
  with respect to neighboring atoms, which is required for neural networks.

.. _lammpssnap: https://docs.lammps.org/compute_sna_atom.html 
.. _quadsnappaper: https://aip.scitation.org/doi/full/10.1063/1.5017641 
.. _chemsnappaper: https://www.sciencedirect.com/science/article/pii/S0021999114008353

[CALCULATOR]
^^^^^^^^^^^^

This section houses keywords determining which calculator to use, i.e. which descriptors to 
calculate. 

- :code:`calculator` is the name of the LAMMPS connection for getting descriptors, e.g. for SNAP 
  descriptors use :code:`LAMMPSSNAP`.

- :code:`energy` is 0 or 1, determining whether to calculate descriptors associated with 
  energies.

- :code:`force` is 0 or 1, determining whether to calculate descriptor gradients 
  associated with forces.

- :code:`stress` is 0 or 1, determining whether to calculate descriptors gradients associated with 
  virial terms for calculating and fitting to stresses.

- :code:`per_atom_energy` is 0 or 1, determining whether to use per-atom energy descriptors in 
  association with :code:`bikflag = 1`

- :code:`nonlinear` is 0 or 1, and should be 1 if using nonlinear solvers such as PyTorch models. 

[ESHIFT]
^^^^^^^^

This section declares an energy shift applied to each atom type.

[SOLVER]
^^^^^^^^

This section contains keywords associated with specific machine learning solvers. 

- :code:`solver` name of the solver. We recommend using :code:`SVD` for linear solvers and 
  :code:`PYTORCH` for neural networks. 

[SCRAPER]
^^^^^^^^^

This section declares which file scraper to use for gathering training data.

- :code:`scraper` is either :code:`JSON` or :code:`XYZ.`

[PATH]
^^^^^^

This section contains a :code:`dataPath` keyword that locates the directory of the training data. 
For example if the training data is in a file called :code:`JSON` in the previous directory relative 
to where we run the FitSNAP executable, this section looks like::

    [PATH]
    dataPath = ../JSON

[OUTFILE]
^^^^^^^^^

This section declares the names of output files.

- :code:`metrics` gives the name of the error metrics markdown file. If using LAMMPS metal units, 
  energy mean absolute errors are in eV and force errors are in eV/Angstrom. 

- :code:`potential` gives the prefix of the LAMMPS-ready potential files to dump.

[REFERENCE]
^^^^^^^^^^^

This section includes settings for an *optional* potential to overlay our machine learned potential 
with. We call this a "reference potential", which is a pair style defined in LAMMPS. If you choose 
to use a reference potential, the energies and forces from the reference potential will be subtracted 
from the target *ab initio* training data. We also declare units in this section.

- :code:`units` declares units used by LAMMPS, see `LAMMPS units docs <lammpsunits_>`_ for more 
  info. 

- :code:`atom_style` the atom style used by the LAMMPS pair style you wish to overlay, see 
  `LAMMPS atom style docs <lammpsatomstyle_>`_ for more info. 

The minimum working reference potential setup involves not using a reference potential at all, where 
the reference section would look like (using metal units)::

    [REFERENCE]
    units = metal
    pair_style = zero 10.0
    pair_coeff = * *

The rest of the keywords are associated with the particular LAMMPS pair style you wish to use. 

.. _lammpsunits: https://docs.lammps.org/units.html
.. _lammpsatomstyle: https://docs.lammps.org/atom_style.html

[GROUPS]
^^^^^^^^

Each group should be its own sub-irectory in the directory given by the :code:`dataPath/` keyword in 
`the [PATH] section <Run.html#path>`__. There are a few different allowed syntaxes; subdirectory 
names in the first column is common to all options.

:code:`group_sections` declares which parameters you want to set for each group of configurations. 

For example::

    group_sections = name training_size testing_size eweight fweight vweight

means you will supply group names, training size as a decimal fraction, testing size as a decimal 
fraction, energy weight, force weight, and virial weight, respectively. We must also declare the 
data types associated with these variables, given by

    group_types = str float float float float float

Then we may declare the group names and parameters associated with them. For a particular group 
called :code:`Liquid` for example, this looks like::

    Liquid        =  1.0    0.0       4.67E+02        1       1.00E-08

where :code:`Liquid` is the name of the group, :code:`1.0` is the training fraction, :code:`0.0` is 
the testing fraction, :code:`6.47E+02` is the energy weight, :code:`1` is the force weight, and 
:code:`1.00E-8` is the virial weight.

Other available keywords are

- :code:`random_sampling` is 0 or 1. If 1, configurations in the groups are randomly sampled between 
  their training and testing fractions. 

- :code:`smartweights`` is 0 or 1. If 1, we declare statistically distributed weights given your 
  supplied weights.

A few examples are found in the examples directory.

[EXTRAS]
^^^^^^^^

This section contains keywords on optional info to dump. By default, linear models output error 
metric markdown files that should be sufficient in most cases. If more detailed errors are required, 
please see the output Pandas dataframe :code:`FitSNAP.df` used by linear models. Examples and 
library tools for analyzing  this dataframe are found in our 
`Colab Python notebook tutorial <tutorialnotebook_>`_.

[MEMORY]
^^^^^^^^

This section contains keywords for dealing with memory. We recommend using defaults. 

Outputs
-------

FitSNAP outputs include error metrics, detailed errors for each atom, configuration, or groups of 
configurations, and LAMMPS-ready files for running MD simulations.

Outputs are different for linear and nonlinear models. 

For linear models, please see the `Linear models output section <Linear.html#outputs>`__

For nonlinear models, please see the
`PyTorch models output section <Pytorch.html#outputs-and-error-calculation>`__. After running a 
linear model fit, the following outputs will be produced:

Library
-------

FitSNAP may also be run through the library interface. The `GitHub repo examples <libexamples_>`_ 
may help set up scripts for your needs. More useful library functionality is found in our 
`Colab Python notebook tutorial <tutorialnotebook_>`_.

.. _libexamples: https://github.com/FitSNAP/FitSNAP/tree/master/examples/library


