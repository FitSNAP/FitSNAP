Usage
=====

There are a few ways to use and install FitSNAP, we'll detail installation below. All options 
involve cloning the GitHub repo:

.. code-block:: console

    git clone https://github.com/FitSNAP/FitSNAP

which creates a :code:`FitSNAP` directory. This directory should be added to your :code:`PYTHONPATH`
, e.g. put this in your :code:`~/.bashrc` (Linux) or :code:`~/.bash_profile` (MacOS)

.. code-block:: console

    export PYTHONPATH="path/to/FitSNAP:$PYTHONPATH"

After that, follow one of the installation options below.

Installation
------------

We can install FitSNAP with a minimal Conda environment, or more manually with our own custom environment.

Minimal Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^

This is explained in the README at https://github.com/FitSNAP/FitSNAP.

This is the easiest way to get started with using FitSNAP, but it does not use the most up-to-date
LAMMPS version which has newer features (e.g. neural networks).

Custom environment with newest LAMMPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the option if you want to use the most up-to-date version of LAMMPS for fitting/using 
neural network potentials or other new features. First we must install LAMMPS and build it for 
FitSNAP. Start with cloning the LAMMPS repo:

.. code-block:: console

    git clone https://github.com/lammps/lammps

Next, make sure you have a Python environment that you want to use FitSNAP in. 

Use :code:`pip` or :code:`conda` to activate your environment before proceeding, and install the
following:

.. code-block:: console

    pip install virtualenv # used to install PyLammps
    pip install torch 
    pip install Cython

Then move into your LAMMPS directory and use cmake to prepare a Makefile that'll build LAMMPS with all the 
packages required by FitSNAP:

.. code-block:: console

    cd lammps
    mkdir build-fitsnap
    cd build-fitsnap
    cmake ../cmake -DLAMMPS_EXCEPTIONS=yes -DBUILD_SHARED_LIBS=yes -DMLIAP_ENABLE_PYTHON=yes -DPKG_PYTHON=yes -DPKG_ML-SNAP=yes -DPKG_ML-IAP=yes -DPKG_ML-PACE=yes -DPKG_SPIN=yes

You might observe some errors, such as missing Cython modules. Install them accordingly with 
:code:`pip` or :code:`conda`. In the same :code:`lammps/build-fitsnap` directory, do

.. code-block:: console

    make
    make install-python

This creates a LAMMPS executable :code:`lmp` that can be used to run FitSNAP potentials, and 
installs PyLammps to your Python's site-packages directory. Check your python installation by doing
the following in your Python interpreter.

>>> from lammps import lammps
>>> lmp = lammps()

This should not produce any errors. If you have errors, please refer to the LAMMPS Python docs:
https://docs.lammps.org/Python_head.html for more instructions on how to install LAMMPS with Python.

After LAMMPS is working in Python, we may proceed to use FitSNAP.