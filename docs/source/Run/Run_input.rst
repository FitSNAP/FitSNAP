Input files
===========

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

This section contains settings for the SNAP bispectrum descriptors from `Thompson et al. (2015) <snappaper_>`_

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
  :code:`2 * rcutfac` to determine the effective cutoff of a particular type. For each element, the 
  effective cutoff radius is :code:`2 * rcutfac * radelem`.

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
.. _chemsnappaper: https://doi.org/10.1021/acs.jpca.0c02450

[ACE]
^^^^^

This section contains settings for the Atomic Cluster Expansion (ACE) descriptors from `Drautz (2019) <drautz2019_>`_ available as `pair_style pace <acelammps_>`_ in LAMMPS. ACE descriptor calculations are explained in `Goff, Sievers, Wood, Thompson (2024) <goff2024_>`_ and with more details on hyperparameters in `Bochkarev et al. <bochkarev2022_>`_ .

.. _drautz2019: https://doi.org/10.1103/PhysRevB.99.014104
.. _pozdnyakov2020: https://doi.org/10.1103/PhysRevLett.125.166001
.. _bochkarev2022: https://doi.org/10.1103/PhysRevMaterials.6.013804
.. _acelammps: https://docs.lammps.org/pair_pace.html
.. _goff2024: https://doi.org/10.1016/j.jcp.2024.113073

- :code:`numTypes` number of atom types in your set of configurations located in `the [PATH] section <Run.html#path>`__

- :code:`type` contains a list of element type symbols, one for each type. Make sure these are 
  ordered correctly, e.g. if you have a LAMMPS type 1 atom that is :code:`Ga`, and LAMMPS type 2 
  atoms are :code:`N`, list this as :code:`Ga N`. In ACE, all possible combinations of these types determine the "bond types" spanned by the ACE chemical basis (the chemical basis is the delta function basis used in `Drautz 2019 <drautz2019_>`_). For example, the "bond types" resulting from all possible combinations of :code:`Ga N` are determined with :code:`itertools.product()` in python. They are :code:`(Ga,Ga) (Ga,N) (N,Ga) (N,N)`. While specifying :code:`types` is sufficient to define the chemical basis in ACE descriptors, some hyperparameters (e.g., :code:`rcutfac`) must be specified per "bond type".

- :code:`ranks` The ranks of the ACE descriptors to be enumerated. Rank is often given the symbol 'N', and corresponds to the number of bonds encoded by the descriptor. Rank 1 corresponds to 1 bond with a central atom, encoding 2-body information, rank 2 corresponds to 2 bonds with a central atom, encoding 3-body information, and so on. The minimum rank should generally be 1 and the maximum rank should be truncated based on the systems modeled and the training dataset, or one will not be able to distinguish certain motifs (see `Pozdnyakov et al. <pozdnyakov2020_>`_ and related works).

- :code:`lmax` is the maximum angular momentum quantum number per descriptor rank. Each ACE descriptors angular character described by one or more spherical harmonics. Since the spherical harmonic basis is infinite, it must be truncated. The maximum angular momentum quantum number (and the truncation) of the spherical harmonic(s) per descriptor rank is specified by :code:`lmax`. Larger :code:`lmax` offer a more complete description of angular character, but lead to larger descriptor counts. For rank 1, :code:`lmax` should not excede 0 for rotationally invariant descriptors. If it is, it will automatically be set to 0. There are no other formal restrictions on :code:`lmax` for :code:`ranks` > 1.

- :code:`lmin` is the minimum angular momentum per descriptor rank. These should be 0 if complete ACE descriptor sets are desired. However, :code:`lmin` may be adjusted up to the :code:`lmax` of the corresponding descriptor rank to reduce the number of descriptors, which can be useful in some neural network fits.

- :code:`nmax` is maximum radial basis function index per descriptor rank. Each ACE descriptors radial character is described by a radial basis. In FitSNAP, the default radial basis set is the Chebyshev polynomial basis described in `Drautz 2019 <drautz2019_>`_. As with the angular basis, the radial basis set is, in principle, infinite and must be truncated.  Larger `nmax` offer a more complete description of radial character, but lead to larger descriptor counts.

- :code:`nmaxbase` Maximum value of `nmax` 

- :code:`rcutfac` *(similar to [BISPECTRUM])* is a cutoff radius parameter or a list of cutoff radius parameters. A value of :code:`rcutfac` must be specified per bond type. For example, if :code:`type` is :code:`Ta`, then one :code:`rcutfac` is specified for the :code:`(Ta,Ta)` bond. If :code:`type` is :code:`Ga N`, then four :code:`rcutfac` must be listed, corresponding to each bond type in :code:`(Ga,Ga) (Ga,N) (N,Ga) (N,N)`. Note that you may want to consider the same :code:`rcutfac` for "bond types" that are permutations of another (e.g.,  for :code:`(Ga,N)` and :code:`(N,Ga)`).

- :code:`lambda` exponential factor for scaled radial distance (see `Drautz 2019 <drautz2019_>`_). The :code:`lambda` parameter(s) determine how much to scale short bond lengths in the scaled distance. The scaled distance is fed into the radial basis for ACE. A larger :code:`lambda` provides more sensitivity for small bond lengths compared to larger bond lengths, and does so through an exponential function. As :code:`lambda` approaches 0, the scaled distance approaches the true distance :code:`r`. As with :code:`rcutfac`, a value of :code:`lambda` must be specified per bond type. For example, if :code:`type` is :code:`Ta`, then one :code:`lambda` is specified for the :code:`(Ta,Ta)` bond. If :code:`type` is :code:`Ga N`, then four :code:`lambda` must be listed, corresponding to each bond type in :code:`(Ga,Ga) (Ga,N) (N,Ga) (N,N)`. Note that you may want to consider the same :code:`lambda` for "bond types" that are permutations of another (e.g.,  for :code:`(Ga,N)` and :code:`(N,Ga)`).

- :code:`rcinner` the cutoff(s) for the core repulsions in the ACE descriptors. A core repulsion may be added to ACE models. As with :code:`rcutfac`, :code:`rcinner` must be provided per "bond type". The default is to set this to 0 for each bond type. Note that, if using :code:`rcinner` values greater than 0.0, the :code:`rcinner` must be selected carefully if using ZBL reference potential(s) in LAMMPS. Typically, one will need to make sure that :code:`rcinner` - :code:`drcinner` are less than the inner cutoffs for the respective ZBL potentials.

- :code:`drcinner` the parameter for the scale of the inner cutoff functions. This is the delta for the inner cutoff cuntion in Eq. C12 in `Bochkarev et al. (2022) <bochkarev2022_>`_. As with :code:`rcutfac`, :code:`rcinner` must be provided per "bond type". The default is to set this to 0.01 per "bond type".

- :code:`bzeroflag` *(same as in [BISPECTRUM])* is 0 or 1, determining whether or not B0, the bispectrum components of an atom with no neighbors, are subtracted from the calculated bispectrum components.

- :code:`wigner_flag` is a logical flag to use generalized Wigner symbols to perform ACE couplings. Generalized wigner symbols or generalized Clebsch-Gordan (CG) coefficients may be used to contract products of spherical harmonics in ACE (usually to obtain rotationally invariant descriptors for MLIPs). Either Wigner symbols or CG coefficients may be used to obtain the correct descriptor couplings, but the numerical scale of the descriptors differs (systematically) when Wigner vs CG are used. The default is to use :code:`wigner_flag` set to :code:`1`, but :code:`wigner_flag` set to :code:`0` will use generalized CG coefficients. If comparing to other LAMMPS potentials generated from other codebases with ACE (e.g, those from `Bochkarev et al. (2022) <bochkarev2022_>`_), using :code:`0` (CG coefficients instead) will help provide a more direct comparison.

- :code:`b_basis` ACE basis flags with possible values `pa_tabulated`, `minsub`, `ysg_x_so3`. Linearly independent descriptors are obtained with `pa_tabulated` as described in <goff2024_>. The `minsub` setting is to maintain compatiblity with descriptor sets from the original <drautz2019_> paper, but it does not give a complete set of ACE descriptors. The `ysg_x_so3` setting is for testing with the overcomplete ACE basis, generated by all possible couplings. It is recommended to use `pa_tabulated` only.


- :code:`manuallabs` manually listed ACE descriptor labels. Optionally, users may specify a list of ACE descriptors to build a model. The ACE descriptor format is mu0_mu1,mu2,...,muN,n1,n2,...,nN,l1,l2,...,lN_L1-...-L(N-3)-L(N-2). For example, to make a model using only one rank 3 descriptor with mu1=mu2=mu3=0 and n1=n2=n3=1 and l1=l2=l3=0 and L1=0, the setting for :code:`manuallabs` could be :code:`0_0,0,0,1,1,1,0,0,0_0 1_0,0,0,1,1,1,0,0,0_0`, where the descriptors differ by mu0. Scripts in the :code:`tools`. Descriptors listed here will override those enumerated by :code:`ranks`, :code:`nmax`, and :code:`lmax`. It can be useful for neural network models, for retraining models only using descriptors with non-zero coefficients from a sparse regression method like LASSO, etc. Note that some functionality within ML-IAP in LAMMPS and others in FitSNAP/LAMMPS require that the number of descriptors per type (per mu0) must be the same. 

.. WARNING:: Only change ACE basis flags if you know what you are doing!

[PYACE]
^^^^^^^

This section allows detailed control over ACE descriptors using the `pyace` Python package, providing compatibility with `pacemaker <https://pacemaker.readthedocs.io/>`_ input formats. This is an alternative to the [ACE] section that provides more flexibility and direct access to pyace functionality.

.. warning::

    [PYACE] is currently EXPERIMENTAL and in development. For production runs, [ACE] is recommended.

The PYACE section supports two input formats:

1. **JSON format** (recommended for complex configurations)
2. **Simple key-value format** (for basic configurations)

.. note::

    Simple format only works for uniform settings across all bonds.

**Basic Settings:**

- :code:`elements` list of element symbols, e.g. :code:`Al Ni` for a binary system

- :code:`cutoff` global cutoff radius for neighbor list construction (Angstroms)

- :code:`delta_spline_bins` spacing for spline interpolation, default 0.001

**JSON Format (Advanced):**

For complex multi-element systems, use JSON strings to specify detailed configurations:

- :code:`embeddings` JSON string specifying embedding functions for each element::

    embeddings = {
      "Al": {
        "npot": "FinnisSinclairShiftedScaled",
        "fs_parameters": [1, 1, 1, 0.5],
        "ndensity": 2,
        "rho_core_cut": 200000,
        "drho_core_cut": 250
      },
      "Ni": {
        "npot": "FinnisSinclairShiftedScaled", 
        "fs_parameters": [1, 1],
        "ndensity": 1,
        "rho_core_cut": 3000,
        "drho_core_cut": 150
      }
    }

- :code:`bonds` JSON string specifying bond parameters for element pairs::

    bonds = {
      "ALL": {
        "radbase": "ChebExpCos",
        "radparameters": [5.25],
        "rcut": 5.0,
        "dcut": 0.01,
        "r_in": 1.0,
        "delta_in": 0.5,
        "core-repulsion": [100.0, 5.0]
      },
      "BINARY": {
        "radbase": "ChebPow",
        "radparameters": [6.25],
        "rcut": 5.5
      }
    }

- :code:`functions` JSON string specifying basis functions::

    functions = {
      "UNARY": {
        "nradmax_by_orders": [15, 3, 2, 2, 1],
        "lmax_by_orders": [0, 2, 2, 1, 1],
        "coefs_init": "zero"
      },
      "BINARY": {
        "nradmax_by_orders": [15, 2, 2, 2],
        "lmax_by_orders": [0, 2, 2, 1]
      }
    }

**Simple Format (Basic):**

For uniform settings across all elements/bonds, use simple key-value pairs:

*Embedding parameters:*

- :code:`embedding_npot` embedding function type, default "FinnisSinclairShiftedScaled"
- :code:`embedding_fs_parameters` JSON list of Finnis-Sinclair parameters, e.g. [1, 1]
- :code:`embedding_ndensity` number of density components, default 1
- :code:`embedding_rho_core_cut` core repulsion density cutoff, default 200000
- :code:`embedding_drho_core_cut` core repulsion density transition width, default 250

*Bond parameters:*

- :code:`bond_radbase` radial basis function type, default "ChebExpCos"
- :code:`bond_radparameters` JSON list of radial parameters, e.g. [5.25]
- :code:`bond_rcut` outer cutoff radius (Angstroms), default 5.0
- :code:`bond_dcut` outer cutoff transition width, default 0.01
- :code:`bond_r_in` inner cutoff radius, default 1.0
- :code:`bond_delta_in` inner cutoff transition width, default 0.5
- :code:`bond_core_repulsion` JSON list [prefactor, lambda] for core repulsion, default [100.0, 5.0]

*Function parameters:*

- :code:`function_nradmax_by_orders` JSON list of max radial basis functions per body order, e.g. [15, 3, 2, 2, 1]
- :code:`function_lmax_by_orders` JSON list of max angular momentum per body order, e.g. [0, 2, 2, 1, 1]
- :code:`function_coefs_init` coefficient initialization: "zero" (default) or "random"

**Legacy Compatibility:**

- :code:`type` alternative to :code:`elements` for backwards compatibility
- :code:`bzeroflag` subtract isolated atom contributions (0 or 1)

**Keywords in JSON configurations:**

In the JSON configurations, you can use keywords to apply settings to groups of elements/bonds:

- :code:`ALL` applies to all element combinations
- :code:`UNARY` applies to single-element (self-interaction) terms
- :code:`BINARY` applies to two-element interaction terms
- :code:`TERNARY` applies to three-element interaction terms
- Element-specific keys like :code:`Al`, :code:`Ni` for individual elements
- Bond-specific keys like :code:`AlAl`, :code:`AlNi` for specific element pairs

More specific keywords override less specific ones (e.g., :code:`AlNi` overrides :code:`BINARY` which overrides :code:`ALL`).

**Example simple configuration:**

.. code-block:: ini

    [PYACE]
    elements = Al Ni
    cutoff = 6.0
    delta_spline_bins = 0.001
    
    # Simple uniform settings
    embedding_npot = FinnisSinclairShiftedScaled
    embedding_fs_parameters = [1, 1]
    embedding_ndensity = 1
    
    bond_radbase = ChebExpCos
    bond_radparameters = [5.25]
    bond_rcut = 5.0
    bond_dcut = 0.01
    
    function_nradmax_by_orders = [15, 3, 2, 2, 1]
    function_lmax_by_orders = [0, 2, 2, 1, 1]

**Example advanced configuration with JSON:**

.. code-block:: ini

    [PYACE]
    elements = Al Ni
    cutoff = 6.0
    delta_spline_bins = 0.001
    
    # Detailed per-element embeddings
    embeddings = {"Al": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}, "Ni": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1, 1], "ndensity": 1}}
    
    # Different bond settings for different pair types  
    bonds = {"ALL": {"radbase": "ChebExpCos", "radparameters": [5.25], "rcut": 5.0}, "BINARY": {"radbase": "ChebPow", "radparameters": [6.25], "rcut": 5.5}}
    
    # Body-order specific function settings
    functions = {"UNARY": {"nradmax_by_orders": [15, 3, 2, 2, 1], "lmax_by_orders": [0, 2, 2, 1, 1]}, "BINARY": {"nradmax_by_orders": [15, 2, 2, 2], "lmax_by_orders": [0, 2, 2, 1]}}

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

This section declares an energy shift applied to each atom type. These values are free to choose however desired. For 
example these values could come from the per-atom energy predicted in a vacuum from *ab initio* calculations. These 
values may also be treated as hyperparameters.

[SOLVER]
^^^^^^^^

This section contains keywords associated with specific machine learning solvers. 

- :code:`solver` name of the solver. We recommend using :code:`SVD` for linear solvers and 
  :code:`PYTORCH` for neural networks. 

[SCRAPER]
^^^^^^^^^

This section declares which file scraper to use for gathering training data.

- :code:`scraper` is either :code:`JSON` or :code:`XYZ.`

If using the XYZ scraper, each `Group <Run.html#groups>`__ of configurations has its own XYZ file 
containing configurations of atoms concatenated together, in extended XYZ format. Follow the example 
in :code:`examples/Ta_XYZ`.

If using the JSON scraper, each `Group <Run.html#groups>`__ may have its own directory containing 
separate JSON files for each configuration. Guarantee compatibility with FitSNAP by using our 
:code:`tools/VASP2JSON.py` conversion script; this requires that your DFT training data be in VASP 
OUTCAR format. Likewise for :code:`tools/VASPxml2JSON.py`.

We are also working on a scraper that directly reads VASP output; more documentation on this coming 
soon.

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

Each group should be its own sub-directory in the directory given by the :code:`dataPath/` keyword in 
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

.. _tutorialnotebook: https://colab.research.google.com/github/FitSNAP/FitSNAP/blob/master/tutorial.ipynb
