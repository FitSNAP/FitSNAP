Usage
=====

Installation
------------

To use FitSNAP, first install it according to the following instructions.

Simple conda install:
^^^^^^^^^^^^^^^^^^^^^
A minimal working environment can be set up using the Python distribution package Anaconda (https://www.anaconda.com).

After installing Anaconda:

#. Clone this repository
.. 
    TODO add clone command
#. Add the cloned repository path to your PYTHONPATH environment variable
..
    TODO add commands for pythonppath
#. Add conda-forge to your Conda install, if not already added::
    
        conda config --add channels conda-forge
#. Create a new Conda environment::

        conda create -n fitsnap python=3.9
        conda activate fitsnap
#. Install packages::

        conda install lammps psutil pandas tabulate sphinx sphinx_rtd_theme mpi4py

WARNING: Conda lammps installation does NOT include ACE descriptor set or SPIN package needed for these corresponding examples.

TIP: Periodically use the command :code:`git pull` in the cloned directory for updates 

