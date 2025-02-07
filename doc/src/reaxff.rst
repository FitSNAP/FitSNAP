ReaxFF Models
=============

Introduction
------------

The `Reactive Force Field (ReaxFF) <https://doi.org/10.1038/npjcompumats.2015.11>`_ replaces fixed bond topologies of classical force fields with the concept of bond order to simulate bond breaking/formation of chemical reactions. Originally conceived for hydrocarbons in the gas phase :footcite:p:`vanduin2001`, ReaxFF has been extended to a wide range of applications :footcite:p:`senftle2016`.


..  youtube:: bmOQ74kkd6A
  :align: center
  :width: 62%

|

.. danger::

  Just because a ReaxFF potential is available with the atoms for your intented application, it **DOES NOT** mean it is transferable if the training set did not include configurations similar to your intented application. For example, there are many potentials with C/H/O/N atoms but not all have the pi-bond parameters trained so a benzene molecule might behave in a *completely unphysical manner*. **You need to consult the original journal article (doi links below) together with the supplementary materials to confirm the transferability of a given ReaxFF potential to your application.**

The Potential Energy Surface (PES) is an insanely immense mathematical object. PES of a system with N atoms doesn't have N points, it has *3N dimensions*! The odds are infinitesimal that someone else visited the same tiny slice of subspace you're interested in and made a Machine-Learning Inter-Atomic-Potential (ML-IAP) or a Reax Force Field (FF) for you already. Stop looking for potentials from somewhere else, except to practice and learn to maybe get close to what you're doing. For *original research* there's no way around having to generate your own DFT and/or experimental data to train a new MLIAP or FF. **This is the purpose of FitSNAP-ReaxFF.**

ReaxFF in LAMMPS :footcite:p:`aktulga2012` supports three charge equilibration methods to represent the dynamics of electron density:

  - Charge Equilibration (QEq) :footcite:p:`rappe1991,nakano1997`

  - Atom-Condensed Kohn-Sham DFT to second order (ACKS2) :footcite:p:`verstraelen2013,ohearn2020`

  - Charge Transfer and Polarization in Equilibrium (QTPIE) :footcite:p:`chen2007`

while fixed partial charges in classical force fields (eg. CHARMM) do not. **FitSNAP-ReaxFF enables retraining of legacy ReaxFF QEq potentials for ACKS2 and QTPIE**, including optimization of the bond_softness, chi, eta, gamma, bcut_acks2, and gauss_exp parameters.


|

--------

ReaxFF functional form
----------------------

The ReaxFF overall system energy is expressed as the sum:

.. math::

  E_{system} & = E_{bond} + E_{lp} + E_{over} + E_{under} + E_{val} + E_{pen} + E_{coa} + E_{C2}\\[.6em]
  & \qquad + E_{triple} + E_{tors} + E_{conj} + E_{Hbond} + E_{vdWaals} + E_{Coulomb}

Details for each term:

- Bond order/energy

- Lone pair energy

- Overcoordination

- Undercoordination

- Valence Terms (Angle energy, Penalty energy, Three-body conjugation term)

- Correction for C2

- Triple bond energy correction

- Torsion Terms (Torsion rotation barriers, Four body conjugation term)

- Hydrogen bond interactions

- Nonbonded interactions (van der Waals, Coulomb)

are presented in the `Supporting Information <https://doi.org/10.1021/jp709896w>`_ of *A ReaxFF Reactive Force Field for Molecular Dynamics Simulations of Hydrocarbon Oxidation* by Chenoweth, van Duin, Goddard (2008).

|

--------

ReaxFF LAMMPS commands
----------------------

* `pair_style reaxff <https://doc.lammps.org/pair_reaxff.html>`_ (/kk)
* `fix qeq/reaxff <https://doc.lammps.org/fix_qeq_reaxff.html>`_ (/kk)
* `fix acks2/reaxff <https://doc.lammps.org/fix_acks2_reaxff.html>`_ (/kk)
* `fix qtpie/reaxff <https://doc.lammps.org/fix_qtpie_reaxff.html>`_ (/kk)
* `compute reaxff/bonds <https://doc.lammps.org/compute_reaxff_bonds.html>`_ (/kk)
* `compute reaxff/species <https://doc.lammps.org/compute_reaxff_species.html>`_ (/kk)
* `compute reaxff/atom <https://doc.lammps.org/compute_reaxff_atom.html>`_ (/kk)

where (/kk) denotes LAMMPS commands available in KOKKOS package.

.. note::

  KOKKOS version of ReaxFF with ``-k on t 1 -sf kk`` is always used by FitSNAP-ReaxFF.

    | *"IMO anyone and everyone should be using the KOKKOS version of ReaxFF. Not only is it more memory robust and will never have these hbondchk errors, it is also faster on CPUs, at least in most cases that I’ve benchmarked, or same speed at the very least."*
    | -- Stan Moore (2024/10) on MatSci.org:
    | **Lammps hbondchk failed**.
    | https://matsci.org/t/lammps-hbondchk-failed/58230/6

    | *"I highly suggest using the KOKKOS package for ReaxFF, works in serial for CPUs too."*
    | -- Stan Moore (2024/10) on MatSci.org:
    | **Segmentation fault: address not mapped to object at address 0xc2cfb87c**.
    | https://matsci.org/t/segmentation-fault-address-not-mapped-to-object-at-address-0xc2cfb87c/58493/5

    | *"You could also try the KOKKOS version which doesn’t use the safezone, mincap, and minhbonds factors which can bloat the memory if you set them too high."*
    | -- Stan Moore (2025/01) on MatSci.org:
    | **Possible memory problem with Reaxff when the total atom number increased**.
    | https://matsci.org/t/possible-memory-problem-with-reaxff-when-the-total-atom-number-increased/60431/2

|

--------

Fitting ReaxFF parameters
-------------------------

If a ReaxFF potential is not available for your intented application, then you can fit new ``parameters`` with FitSNAP-ReaxFF from DFT training data. FitSNAP-ReaxFF is based on the `Covariance Matrix Adaptation Evolution Strategy (CMAES) <http://cma-es.github.io/>`_ optimization algorithm as implemented by the `pycma python package <https://github.com/CMA-ES/pycma>`_. CMAES finds a minimum :math:`x \in \mathbb{R}^n` of an objective function :math:`f(x)`. In FitSNAP-ReaxFF, the objective function minimized is the Sum of Squared Errors (SSE) between DFT reference data and predicted energy/forces given current values of parameters to be optimized.

The FitSNAP-ReaxFF workflow is fundamentally different than FitSNAP but relies on the same underlying infrastructure:

**FitSNAP (SNAP/PACE/...)**
  Two separate phases after scraping data: (i) *process_configs()* to calculate descriptors and (ii) *perform_fit()* to solve for optimal coefficients.

**FitSNAP-ReaxFF**
  One integrated phase: *perform_fit()* consists of a loop where *process_configs()* runs in parallel at each step of the fitting algorithm. During this loop, a population of ``popsize`` candidate ``parameters`` is refined until the CMAES algorithm meets a termination criteria.

You can start a FitSNAP-ReaxFF optimization with a potential file from   ``reaxff/potentials/reaxff-<AUTHOR><YEAR>.ff`` :ref:`(see below for full list bundled with FitSNAP-ReaxFF) <available_potentials>`. You can also start with any other valid ReaxFF potential file (with the exception of *eReaxFF* and *LG dispersion correction*), or :guilabel:`FIXME: restart from a previously optimized potential`.

.. admonition:: N2_ReaxFF example
  :class: Hint

  Let's start with a simple example related to the `nitrogen molecule example <https://alphataubio.com/inq/tutorial_shell_python.html>`_ of INQ, a modern clean-slate C++/CUDA open source (TD)DFT package from LLNL. DFT reference data can also be obtained from  `Quantum Espresso (QE) <https://www.quantum-espresso.org/>`_, `Vienna Ab initio Simulation Package (VASP) <https://www.vasp.at/>`_, literature, online databases,...

  *First*, training data is computed using INQ with PBE functional and saved to ``JSON/N2/N2*.json``:

  .. literalinclude:: ../../examples/N2_ReaxFF/N2_ReaxFF-bond-scan.py
    :caption: **examples/N2_ReaxFF/N2_ReaxFF-bond-scan.py**

  *Second*, a FitSNAP-ReaxFF optimization with input scripts ``N2_ReaxFF-<CHARGE_FIX>.in``:

  .. tabs::

   .. tab:: QEQ

      .. literalinclude:: ../../examples/N2_ReaxFF/N2_ReaxFF-qeq.in
        :caption: **examples/N2_ReaxFF/N2_ReaxFF-qeq.in**

   .. tab:: ACKS2

      .. literalinclude:: ../../examples/N2_ReaxFF/N2_ReaxFF-acks2.in
        :caption: **examples/N2_ReaxFF/N2_ReaxFF.in**

   .. tab:: QTPIE

      .. literalinclude:: ../../examples/N2_ReaxFF/N2_ReaxFF-qtpie.in
        :caption: **examples/N2_ReaxFF/N2_ReaxFF-qtpie.in**

  *Third*, potential energy computed along the bond scan :math:`\text{N}\!\equiv\!\text{N}` by running LAMMPS with potentials

    - ``reaxff-wood2014.ff``
    - ``reaxff-N2_ReaxFF-qeq.ff``
    - ``reaxff-N2_ReaxFF-acks2.ff``
    - ``reaxff-N2_ReaxFF-qtpie.ff``

  is compared to QM training data with matplotlib and saved to ``N2_ReaxFF.png``:

  .. image:: ../../examples/N2_ReaxFF/N2_ReaxFF.png
    :align: center
    :width: 62%

FitSNAP-ReaxFF input script
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compared to linear and nonlinear models, the input script for ReaxFF models needs:

  - ``[REAXFF]`` section instead of ``[BISPECTRUM]`` or ``[ACE]`` section

  - ``calculator = LAMMPSREAXFF`` instead of ``LAMMPSSNAP``, ``LAMMPSPACE``, ...

  - ``solver = CMAES`` instead of eg. ``SVD``, ``PYTORCH``, ...

``[REAXFF]`` section
""""""""""""""""""""

  - ``potential`` path of initial ReaxFF potential file

  - ``parameters`` strings separated by spaces with format ``<BLOCK>.<ATOM_1>...<ATOM_N>.<NAME>``:

      - ``GEN.name`` for atom parameters
      - ``ATM.C.name`` for atom parameters
      - ``BND.C.H.name`` for bond parameters
      - ``OFD.C.H.name`` for off-diagonal parameters
      - ``ANG.C.H.O.name`` for angle parameters
      - ``TOR.C.H.O.N.name`` for torsion parameters
      - ``HBD.C.H.O.name`` for hydrogen-bond parameters

    where ``name`` is *LAMMPS implementation parameter name* (which might be different than other ReaxFF implementations commonly seen in comments of potential files)

    .. - ``'range'`` **optional** python array of two floats to specify minimum and maximum allowed values for a parameter :math:`p`, with default range :math:`p_0\pm.2|p_0|` if :math:`|p_0|>0` and :math:`(-1,1)` otherwise

.. raw:: html
  :file: parameters.html


.. note::

  ``reaxff/tools/reaxff-format-ff.py`` properly reformats a ReaxFF potential file (eg. copy/pasted from journal articles) together with *LAMMPS implementation parameter names* in comment fields.



``[CALCULATOR]`` section
""""""""""""""""""""""""

  - ``calculator`` **must be** ``LAMMPSREAXFF`` **for FitSNAP-ReaxFF**

  - ``charge_fix`` charge equilibration fix command, eg:

    - *(a)* ``fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff``

    - *(b)* ``fix 1 all acks2/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 500``

    - *(c)* ``fix 1 all qtpie/reaxff 1 0.0 10.0 1.0e-6 reaxff exp.qtpie``

    - fix ID (``1`` in *examples a-c*), can only contain alphanumeric characters and underscores to be valid in LAMMPS

  - ``energy`` turn on ``1`` or off ``0`` energy fitting

  - ``force`` turn on ``1`` or off ``0`` force fitting

  - ``stress`` **ignored in FitSNAP-ReaxFF**

  - ``dipole`` turn on ``1`` or off ``0`` dipole fitting

.. note::

  Stress fitting is not supported in FitSNAP-ReaxFF, only ``energy = 1`` and ``force = 1`` are available.


``[SOLVER]`` section
""""""""""""""""""""

  - ``solver`` **must be** ``CMAES`` **for FitSNAP-ReaxFF**

  - ``popsize`` population size setting of CMAES algorithm, with default :math:`4+3*log(|P|)` where :math:`|P|` is the number of parameters to be optimized. [`detailed discussion with the author of the pycma python package <https://github.com/CMA-ES/pycma/issues/140>`_]

  - ``sigma`` sigma setting of CMAES algorithm, with default 0.1


``[SCRAPER]`` section
"""""""""""""""""""""

  - same as FitSNAP


``[PATH]`` section
""""""""""""""""""

  - same as FitSNAP

``[OUTFILE]`` section
"""""""""""""""""""""

  - ``potential`` path of optimized ReaxFF potential file

  - ``output_style`` **not applicable because** ``output_style=REAXFF`` **implied by REAXFF section**


``[REFERENCE]`` section
"""""""""""""""""""""""

  - **not applicable in FitSNAP-ReaxFF**

.. note:: FitSNAP-ReaxFF only supports ``units real`` and ``atom_style charge``.


``[GROUPS]`` section
""""""""""""""""""""

  - same as FitSNAP



--------

.. _available_potentials:

Available ReaxFF potentials
---------------------------

.. list-table:: Historical serial Fortran 77 force fields (no longer compatible and not available)
   :widths: 10 10 10 70
   :header-rows: 1
   :align: center

   * - Branch
     - Atoms
     - Filename
     - Source
   * - combustion
     - C / H
     - *n/a*
     - :footcite:t:`vanduin2001`

Combustion Branch
^^^^^^^^^^^^^^^^^

.. list-table:: Available COMBUSTION force fields in LAMMPS
   :widths: 10 10 10 10 60
   :header-rows: 1
   :align: center

   * - Branch
     - Atoms
     - Filename (LAMMPS)
     - `Filename (SCM) <https://www.scm.com/doc/ReaxFF/Included_Forcefields.html>`_
     - Source
   * - combustion
     - Au/S/C/H
     - reaxff-jarvi2011.ff
     - AuSCH_2011.ff
     - :footcite:t:`jarvi2011`
   * - combustion
     - C
     - reaxff-srinivasan2015.ff
     - C.ff
     - :footcite:t:`srinivasan2015`
   * - combustion
     - C/H
     - reaxff-mao2017.ff
     - CH_aromatics.ff
     - :footcite:t:`mao2017`
   * - combustion
     - C/H/B/N
     - reaxff-pai2016.ff
     - CBN.ff
     - :footcite:t:`pai2016`
   * - combustion
     - C/H/Na
     - reaxff-hjertenaes2016.ff
     - CHNa.ff
     - :footcite:t:`hjertenaes2016`
   * - combustion
     - C/H/O
     - reaxff-ashraf2017.ff
     - CHO-2016.ff
     - :footcite:t:`ashraf2017`
   * - combustion
     - C/H/O
     - reaxff-chenoweth2008a.ff
     - CHO.ff
     - :footcite:t:`chenoweth2008a`
   * - combustion
     - C/H/O/Ba/Zr/Y
     - reaxff-vanduin2008.ff
     - BaYZrCHO.ff
     - :footcite:t:`vanduin2008`
   * - combustion
     - C/H/O/N
     - reaxff-strachan2003.ff
     - *n/a*
     - :footcite:t:`strachan2003`
   * - :guilabel:`FIXME`
     - C/H/O/N
     - reaxff-budzien2009.ff
     - *n/a*
     - :footcite:t:`budzien2009`
   * - :guilabel:`FIXME`
     - C/H/O/N/S
     - reaxff-mattsson2010.ff
     - *n/a*
     - :footcite:t:`mattsson2010`
   * - :guilabel:`FIXME`
     - C/H/O/N/S/F/Pt/Cl/Ni/X
     - reaxff-singh2013.ff
     - *n/a*
     - :footcite:t:`singh2013`
   * - combustion
     - C/H/O/N/S/Si
     - reaxff-liu2011.ff
     - dispersion/CHONSSi-lg.ff
     - :footcite:t:`liu2011`
   * - combustion
     - C/H/O/N/S/Si
     - reaxff-zhang2009.ff
     - HE2.ff
     - :footcite:t:`zhang2009`
   * - combustion
     - C/H/O/N/S/Si/Ge
     - reaxff-psofogiannakis2016.ff
     - CHONSSiGe.ff
     - :footcite:t:`psofogiannakis2016`
   * - combustion
     - C/H/O/N/S/Si/Na/P
     - reaxff-zhang2014.ff
     - CHONSSiNaP.ff
     - :footcite:t:`zhang2014`
   * - combustion
     - C/H/O/N/S/Si/Pt/Zr/Ni/Cu/Co
     - reaxff-nielson2005.ff
     - CHONSSiPtZrNiCuCo.ff
     - :footcite:t:`nielson2005`
   * - combustion
     - C/H/O/N/S/Si/Pt/Ni/Cu/Co/Zr/Y/Ba
     - reaxff-merinov2014.ff
     - CHONSSiPtNiCuCoZrYBa.ff
     - :footcite:t:`merinov2014`
   * - combustion
     - | C/H/O/N/S/Si/Pt/Zr/Ni/
       | Cu/Co/He/Ne/Ar/Kr/Xe
     - reaxff-kamat2010.ff
     - CHONSSiPtZrNiCuCoHeNeArKrXe.ff
     - :footcite:t:`kamat2010`
   * - combustion
     - C/H/O/N/Si/S
     - reaxff-kulkarni2013.ff
     - SiONH.ff
     - :footcite:t:`kulkarni2013`
   * - combustion
     - C/H/O/S
     - reaxff-mueller2016.ff
     - Mue2016.ff
     - :footcite:t:`mueller2016`
   * - combustion
     - C/H/O/S
     - reaxff-komissarov2021.ff
     - *n/a*
     - :footcite:t:`komissarov2021`
   * - combustion
     - C/H/O/S/F/Cl/N
     - reaxff-wood2014.ff
     - CHOSFClN.ff
     - :footcite:t:`wood2014`
   * - combustion
     - C/H/Pt
     - reaxff-sanz2008.ff
     - PtCH.ff
     - :footcite:t:`sanz2008`
   * - combustion
     - C/H/O/Si
     - reaxff-chenoweth2005.ff
     - PDMSDecomp.ff
     - :footcite:t:`chenoweth2005`
   * - :guilabel:`FIXME`
     - H/O/Au
     - reaxff-joshi2010.ff
     - *n/a*
     - :footcite:t:`joshi2010`
   * - combustion
     - Co
     - reaxff-zhang2014b.ff
     - Co.ff
     - :footcite:t:`zhang2014b`
   * - combustion
     - H/O/N/B
     - reaxff-weismiller2010.ff
     - Ab.ff
     - :footcite:t:`weismiller2010`
   * - combustion
     - Li/S
     - reaxff-islam2015.ff
     - LiS.ff
     - :footcite:t:`islam2015`
   * - combustion
     - Ni/C/H
     - reaxff-mueller2010.ff
     - NiCH.ff
     - :footcite:t:`mueller2010`
   * - combustion
     - O/Pt
     - reaxff-fantauzzi2014.ff
     - OPt.ff
     - :footcite:t:`fantauzzi2014`
   * - combustion
     - Pd/H
     - reaxff-senftle2014.ff
     - PdH.ff
     - :footcite:t:`senftle2014`
   * - combustion
     - Si/C/O/H/N/S
     - reaxff-newsome2012.ff
     - SiC.ff
     - :footcite:t:`newsome2012`
   * - combustion
     - V/O/C/H
     - reaxff-chenoweth2008b.ff
     - VOCH.ff
     - :footcite:t:`chenoweth2008b`



Independent Branch
^^^^^^^^^^^^^^^^^^

.. list-table:: Available INDEPENDENT force fields in LAMMPS
   :widths: 10 10 10 10 60
   :header-rows: 1
   :align: center

   * - Branch
     - Atoms
     - Filename (LAMMPS)
     - `Filename (SCM) <https://www.scm.com/doc/ReaxFF/Included_Forcefields.html>`_
     - Source
   * - independent
     - C/H/Ar/He/Ne/Kr
     - reaxff-yoon2016.ff
     - CHArHeNeKr.ff
     - :footcite:t:`yoon2016`
   * - independent
     - C/H/Fe
     - reaxff-islam2016.ff
     - CHFe.ff
     - :footcite:t:`islam2016`
   * - independent
     - | C/H/Ga
       | C/H/In
     - | reaxff-rajabpour2021a.ff
       | reaxff-rajabpour2021b.ff
     - | GaCH-2020.ff
       | InCH-2020.ff
     - :footcite:t:`rajabpour2021`
   * - independent
     - C/H/O/Ge
     - reaxff-nayir2018.ff
     - CHOGe.ff
     - :footcite:t:`nayir2018`
   * - independent
     - C/H/O/Li/Al/Ti/P
     - reaxff-shin2018.ff
     - CHOLiAlTiP.ff
     - :footcite:t:`shin2018`
   * - independent
     - C/H/O/N/B/Al/Si/Cl
     - reaxff-uene2024.ff
     - CHONBAlSiCl.ff
     - :footcite:t:`uene2024`
   * - independent
     - C/H/O/N/S/Mg/P/Na/Cu/Cl/Ti/X
     - reaxff-hou2022.ff
     - CHONSMgPNaCuClTi.ff
     - :footcite:t:`hou2022`
   * - independent
     - C/H/O/N/S/Si
     - reaxff-soria2018.ff
     - CHONSSi.ff
     - :footcite:t:`soria2018`
   * - independent
     - C/H/O/N/S/Si/Ge/Ga/Ag
     - reaxff-niefind2024.ff
     - CHONSSiGeGaAg.ff
     - :footcite:t:`niefind2024`
   * - independent
     - C/H/O/N/S/Zr
     - reaxff-dwivedi2020.ff
     - CHONSZr.ff
     - :footcite:t:`dwivedi2020`
   * - independent
     - C/H/O/N/Si
     - reaxff-wang2020.ff
     - CHONSi.ff
     - :footcite:t:`wang2020`
   * - independent
     - C/H/O/S/Cu/Cl/X
     - reaxff-yeon2018.ff
     - CuSCH.ff
     - :footcite:t:`yeon2018`
   * - independent
     - C/H/O/S/Mo/Ni/Au/Ti
     - reaxff-mao2022.ff
     - CHOSMoNiAuTi.ff
     - :footcite:t:`mao2022`
   * - independent
     - Cu/Zr
     - reaxff-huang2019.ff
     - CuZr.ff
     - :footcite:t:`huang2019`
   * - independent
     - H/O/N/Si/F
     - reaxff-kim2021.ff
     - HONSiF.ff
     - :footcite:t:`kim2021`
   * - independent
     - H/O/Si/Al/Li
     - reaxff-ostadhossein2016.ff
     - HOSiAlLi.ff
     - :footcite:t:`ostadhossein2016`
   * - independent
     - H/S/Mo
     - reaxff-ostadhossein2017.ff
     - HSMo.ff
     - :footcite:t:`ostadhossein2017`
   * - independent
     - I/Br/Pb/Cs
     - reaxff-pols2024.ff
     - IBrPbCs.ff
     - :footcite:t:`pols2024`
   * - independent
     - I/Pb/Cs/X
     - reaxff-pols2021.ff
     - CsPbI.ff
     - :footcite:t:`pols2021`
   * - independent
     - Li/Si/C
     - reaxff-olou2023.ff
     - LiSiC.ff
     - :footcite:t:`olou2023`
   * - independent
     - Mg/O
     - reaxff-fiesinger2023.ff
     - MgO.ff
     - :footcite:t:`fiesinger2023`
   * - independent
     - Ni/Al
     - reaxff-du2023.ff
     - NiAl.ff
     - :footcite:t:`du2023`
   * - independent
     - Ni/Cr
     - reaxff-shin2021.ff
     - NiCr.ff
     - :footcite:t:`shin2021`
   * - independent
     - Ru/H
     - reaxff-onwudinanti2022.ff
     - RuH.ff
     - :footcite:t:`onwudinanti2022`
   * - independent
     - Ru/N/H
     - reaxff-kim2018.ff
     - RuNH.ff
     - :footcite:t:`kim2018`
   * - independent
     - Si/Al/Mg/O
     - reaxff-yeon2021.ff
     - SiAlMgO.ff
     - :footcite:t:`yeon2021`
   * - independent
     - Si/O/H
     - reaxff-nayir2019.ff
     - SiOHv2.ff
     - :footcite:t:`nayir2019`
   * - independent
     - W/S/H/Al/O
     - reaxff-nayir2021.ff
     - WSHAlO.ff
     - :footcite:t:`nayir2021`
   * - independent
     - Zr/Y/O/H
     - reaxff-mayernick2010.ff
     - ZrYOHVac.ff
     - :footcite:t:`mayernick2010`
   * - independent
     - Zr/Y/O/Ni/H
     - reaxff-liu2019.ff
     - ZrYONiH.ff
     - :footcite:t:`liu2019`




Water Branch
^^^^^^^^^^^^

.. list-table:: Available WATER force fields in LAMMPS
   :widths: 10 10 10 10 60
   :header-rows: 1
   :align: center

   * - Branch
     - Atoms
     - Filename (LAMMPS)
     - `Filename (SCM) <https://www.scm.com/doc/ReaxFF/Included_Forcefields.html>`_
     - Source
   * - water
     - Al/C/H/O
     - reaxff-hong2016.ff
     - AlCHO.ff
     - :footcite:t:`hong2016`
   * - water
     - C/H/O/Al/Ge/X
     - reaxff-zheng2017.ff
     - CHOAlGeX.ff
     - :footcite:t:`zheng2017`
   * - water
     - C/H/O/Ca/Si/X
     - reaxff-manzano2012.ff
     - CaSiOH.ff
     - :footcite:t:`manzano2012`
   * - water
     - C/H/O/Cs/K/Na/Cl/I/F/Li
     - reaxff-fedkin2019.ff
     - CHOCsKNaClIFLi.ff
     - :footcite:t:`fedkin2019`
   * - water
     - C/H/O/Fe
     - reaxff-aryanpour2010.ff
     - FeOCHCl.ff
     - :footcite:t:`aryanpour2010`
   * - water
     - C/H/O/Fe/Al/Ni/Cu/S/Cr
     - reaxff-shin2015.ff
     - CHOFeAlNiCuSCr.ff
     - :footcite:t:`shin2015`
   * - water
     - C/H/O/Fe/Al/Ni/Cu/S/Cr
     - reaxff-tavazza2015.ff
     - CHOFeAlNiCuSCr_v3.ff
     - :footcite:t:`tavazza2015`
   * - water
     - C/H/O/N
     - reaxff-rahaman2011.ff
     - Glycine.ff
     - :footcite:t:`rahaman2011`
   * - water
     - C/H/O/N
     - reaxff-trnka2018.ff
     - *n/a*
     - :footcite:t:`trnka2018`
   * - water
     - C/H/O/N
     - reaxff-kowalik2019.ff
     - CHON-2019.ff
     - :footcite:t:`kowalik2019`
   * - water
     - C/H/O/N/S/Fe
     - reaxff-moerman2021.ff
     - CHONSFe.ff
     - :footcite:t:`moerman2021`
   * - water
     - C/H/O/N/S/Mg/P/Na/Cu
     - reaxff-huang2013.ff
     - CuBTC.ff
     - :footcite:t:`huang2013`
   * - water
     - C/H/O/N/S/Mg/P/Na/Cu/Cl
     - reaxff-monti2013a.ff
     - CHONSMgPNaCuCl.ff
     - :footcite:t:`monti2013a`
   * - water
     - C/H/O/N/S/Mg/P/Na/Cu/Cl
     - reaxff-monti2013b.ff
     - CHONSMgPNaCuCl_v2.ff
     - :footcite:t:`monti2013b`
   * - water
     - C/H/O/N/S/Mg/P/Na/Cu/Cl/X
     - reaxff-zhang2018.ff
     - CHON2017_weak.ff
     - :footcite:t:`zhang2018`
   * - water
     - C/H/O/N/S/Mg/P/Na/Ti/Cl/F
     - reaxff-huygh2014.ff
     - CHONSMgPNaTiClF.ff
     - :footcite:t:`huygh2014`
   * - water
     - C/H/O/N/S/Mg/P/Na/Ti/Cl/F
     - reaxff-kim2013a.ff
     - TiOCHNCl.ff
     - :footcite:t:`kim2013a`
   * - water
     - C/H/O/N/S/Mg/P/Na/Ti/Cl/F
     - reaxff-kim2013b.ff
     - TiClOH.ff
     - :footcite:t:`kim2013b`
   * - water
     - C/H/O/N/S/Mg/P/Na/Ti/Cl/F/Au
     - reaxff-monti2016.ff
     - CHONSMgPNaTiClFAu.ff
     - :footcite:t:`monti2016`
   * - water
     - C/H/O/N/S/Mg/P/Na/Ti/Cl/F/K/Li
     - reaxff-ganeshan2020.ff
     - CHONSMgPNaTiClFKLi.ff
     - :footcite:t:`ganeshan2020`
   * - water
     - C/H/O/N/Si/Cu/Ag/Zn
     - reaxff-lloyd2016.ff
     - AgZnO.ff
     - :footcite:t:`lloyd2016`
   * - water
     - C/H/O/N/S/Si/Ca/Cs/K/Sr/Na/Mg/Al/Cu
     - reaxff-psofogiannakis2015.ff
     - CHONSSiCaCsKSrNaMgAlCu.ff
     - :footcite:t:`psofogiannakis2015`
   * - water
     - C/H/O/N/S/Si/Na/Al
     - reaxff-bai2012.ff
     - CHONSSiNaAl.ff
     - :footcite:t:`bai2012`
   * - water
     - C/H/O/S/Mo/Ni/Li/B/F/P/N
     - reaxff-liu2021.ff
     - CHOSMoNiLiBFPN-2.ff
     - :footcite:t:`liu2021`
   * - water
     - C/H/O/Si/Na
     - reaxff-hahn2018.ff
     - CHOSiNa.ff
     - :footcite:t:`hahn2018`
   * - water
     - C/H/O/Zn
     - reaxff-han2010.ff
     - CHOZn.ff
     - :footcite:t:`han2010`
   * - water
     - H/O/Si/Al/Li
     - reaxff-narayanan2011.ff
     - SiOAlLi.ff
     - :footcite:t:`narayanan2011`
   * - water
     - H/O/X
     - reaxff-zhang2017.ff
     - Water2017.ff
     - :footcite:t:`zhang2017`
   * - water
     - Zn/O/H
     - reaxff-raymand2010.ff
     - ZnOH.ff
     - :footcite:t:`raymand2010`




--------

ReaxFF Bibliography
-------------------

  :download:`download reaxff.bib<reaxff.bib>`

.. footbibliography::

