Introduction
============

Overview of FitSNAP
-------------------

FitSNAP is a machine learning software that focuses on modelling how interatomic interactions, or
potential energy, depends on geometry between atoms. In this sense the machine learning problem is
learning the mapping between atomic coordinates and potential energy. Since the physics of 
interatomic interactions are invariant to translation, rotation, and permutation, we must describe
interatomic geometry in such a way that these symmetries are satisfied. We do this using invariant
"descriptors" such as `SNAP descriptors <snap_>`_, and others. 

The resulting machine learned models therefore predict interatomic interaction energies as a 
function of these geometric descriptors. Once these models are obtained from FitSNAP, they may be
used directly in `LAMMPS <lammps_>`_ to perform high-performance molecular dynamics simulations.

.. _snap: https://www.sciencedirect.com/science/article/pii/S0021999114008353
.. _lammps: https://docs.lammps.org/
