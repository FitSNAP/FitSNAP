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

FitSNAP Lingo
-------------

Throughout these docs we will use phrases or language particular to our general way of doing
molecular machine learning. These phrases are also littered throughout our code architecture, for 
example in the names of our Python classes. To solve molecular machine learning problems, we first
need to **scrape** configurations of atoms from a dataset; this is achieved in the :code:`Scraper` 
class. When storing configurations for fitting, it is useful to organize them into **groups** which 
may have different training/testing ratios or energy/force/stress weights.

Then we need a **calculator** to process these configurations by calculating descriptors for 
all atoms; this is done in the :code:`Calculator` class. Once we have descriptors, we must learn the 
mapping between descriptors and properties like energies and forces with a **solver** that performs 
a fit via linear regression or gradient descent for neural networks; these solvers are contained in 
the :code:`Solver` class.