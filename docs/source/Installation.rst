Installation
============

This page documents how to properly install LAMMPS and FitSNAP. First we begin with how to install 
LAMMPS specifically for FitSNAP. 

- If you do not want to manually install LAMMPS, please see `Minimal conda install`_, but note this 
  version of conda LAMMPS does not included recent features like neural networks or ACE descriptors.

- If you want a quick summary of installation instructions, see `Quick Installation <Quick.html>`__.

.. _LAMMPS Installation:

LAMMPS Installation
-------------------

Since LAMMPS is the backbone of FitSNAP, we begin with instructions on how to install LAMMPS 
specifically for using FitSNAP. The following few sections cover basics of installing LAMMPS with 
Python library support. 

- If you want to fit ACE potentials, see `LAMMPS PACE install`_

MPI for LAMMPS and FitSNAP
^^^^^^^^^^^^^^^^^^^^^^^^^^

Both LAMMPS and FitSNAP are parallelized for optimal performance.

We build MPI from source using OpenMPI version 4.1.4 (https://www.open-mpi.org/) 
and the instructions at https://www.open-mpi.org/faq/?category=building#easy-build.

After building, add your openmpi executable path to your :code:`PATH` variable as well so that 
LAMMPS can automatically find your MPI install, e.g.::
                
        # in e.g., ~/.bashrc
        MPI_DIR=/usr/local/openmpi     
        export PATH=$MPI_DIR/bin:$PATH

Python dependencies
^^^^^^^^^^^^^^^^^^^

We recommend creating a virtual environment with :code:`python -m venv` or :code:`conda`. After 
creating your virtual environment, **make sure it is activated for all future steps**, e.g.::

    conda create --name fitsnap python=3.10
    conda activate fitsnap

Now install the necessary pre-requisites to build Python-LAMMPS using pip or conda::

    python -m pip install numpy scipy scikit-learn virtualenv psutil pandas tabulate mpi4py Cython
    # For nonlinear fitting:
    python -m pip install torch
    # For fitting ACE:
    python -m pip install sympy pyyaml
    # For contributing to docs:
    python -m pip install sphinx sphinx_rtd_theme sphinxcontrib-napoleon

To make sure MPI is working, make a Python script called :code:`test.py` with the following::

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print("Proc %d out of %d procs" % (comm.Get_rank(),comm.Get_size()))

And see the output of running :code:`test.py` in parallel::

    # NOTE: the line order is not deterministic
    $ mpirun -np 4 python test.py
    Proc 0 out of 4 procs
    Proc 1 out of 4 procs
    Proc 2 out of 4 procs
    Proc 3 out of 4 procs

LAMMPS for FitSNAP
^^^^^^^^^^^^^^^^^^

- If you want to fit ACE potentials, see `LAMMPS PACE install`_

First clone the LAMMPS repo::

    git clone https://github.com/lammps/lammps

This creates a :code:`lammps` directory, where we will build LAMMPS using `cmake`` and `make`::

    cd /path/to/lammps
    mkdir build-fitsnap
    cd build-fitsnap
    # Use cmake to build the Makefile
    cmake ../cmake -DLAMMPS_EXCEPTIONS=yes \
                  -DBUILD_SHARED_LIBS=yes \
                  -DMLIAP_ENABLE_PYTHON=yes \
                  -DPKG_PYTHON=yes \
                  -DPKG_ML-SNAP=yes \
                  -DPKG_ML-IAP=yes \
                  -DPKG_ML-PACE=yes \
                  -DPKG_SPIN=yes \
                  -DPYTHON_EXECUTABLE:FILEPATH=`which python`
    # Build a LAMMPS executable and shared library
    make
    # Install Python-LAMMPS interface
    make install-python

Do not be alarmed by runtime library warnings after `cmake`, or `-Weffc++` and `-Wunused-result` 
warnings during `make`.
This will create a LAMMPS executable :code:`lmp`, which should be used to run MD using FitSNAP fits.
This will also create a Python-LAMMPS interface located in your Python :code:`site-packages/lammps` 
directory. Set the following environment variables so that your Python can find the LAMMPS library::

    LAMMPS_DIR=/path/to/lammps
    # For MacOS, use DYLD_LIBRARY_PATH instead of LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LAMMPS_DIR/build-fitsnap:$LD_LIBRARY_PATH 

Also make sure your Python-LAMMPS interface is working by doing the following in your Python 
interpreter::

    import lammps
    lmp = lammps.lammps()

which should produce no errors.

After completing this LAMMPS installation, please see `Install FitSNAP with latest LAMMPS`_ to use 
FitSNAP.

.. _LAMMPS PACE install:

LAMMPS PACE install
^^^^^^^^^^^^^^^^^^^

Computes for ACE descriptors are currently in our modified LAMMPS repo (https://github.com/jmgoff/lammps_compute_PACE), 
so the installation instructions are a little different if you want to use ACE. 

#. Clone our modified LAMMPS repo and set up a typical LAMMPS build::

        git clone -b compute-pace https://github.com/jmgoff/lammps_compute_PACE
        cd lammps_compute_PACE
        mkdir build && cd build

#. Set up a typical LAMMPS build the ML-PACE library enabled::

        cmake ../cmake -DLAMMPS_EXCEPTIONS=yes \
                       -DBUILD_SHARED_LIBS=yes \
                       -DMLIAP_ENABLE_PYTHON=yes \
                       -DPKG_PYTHON=yes \
                       -DPKG_ML-SNAP=yes \
                       -DPKG_ML-IAP=yes \
                       -DPKG_ML-PACE=yes \
                       -DPKG_SPIN=yes \
                       -DPYTHON_EXECUTABLE:FILEPATH=`which python`

#. Download the modified lammps-user-pace code that contains extra arrays for breaking out descriptor contributions::

        git clone https://github.com/jmgoff/lammps-user-pace-1
        cp lammps-user-pace-1/ML-PACE/ace-evaluator/ace_evaluator.* ./lammps-user-pace-v.2022.10.15/ML-PACE/ace-evaluator/
        make -j
        make install


#. Now, set up paths::

        # Use DYLD_LIBRARY_PATH if using MacOS, on Linux use LD_LIBRARY_PATH:
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lammps_compute_PACE/build

#. Now we can get and use FitSNAP::

        cd /path/to/where/you/want/FitSNAP
        git clone https://github.com/FitSNAP/FitSNAP
        # Set python path so you can run FitSNAP as executable:
        export PYTHONPATH=$PYTHONPATH:/path/to/where/you/want/FitSNAP

For a summary/review of all these steps, see see `Quick Installation <Quick.html>`__. 

FitSNAP Installation
--------------------

There are two primary ways to get started with FitSNAP: (1) building LAMMPS manually, and (2) a 
simple conda environment using the packaged LAMMPS that ships with conda. The former option allows 
for more recent LAMMPS features. 

.. _Install FitSNAP with latest LAMMPS:

Install FitSNAP with latest LAMMPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both FitSNAP and LAMMPS have been optimized to work with MPI. For optimal performance of 
both, we recommend building and configuring your favored flavor of MPI before continuing 
(see `LAMMPS Installation`_ docs).

#. Set up environment and build LAMMPS (see `LAMMPS Installation`_ docs)

#. Clone the FitSNAP repository::

        git clone https://github.com/FitSNAP/FitSNAP

#. Add the cloned repository path to your PYTHONPATH environment variable::
        
        FITSNAP_DIR=\path\to\FitSNAP
        export PYTHONPATH=$FITSNAP_DIR:$PYTHONPATH

#. You should now be able to run the FitSNAP examples in :code:`FitSNAP/examples`.

#. For a summary/review of all these steps, see see `Quick Installation <Quick.html>`__. 

.. _Minimal conda install:

Minimal conda install
^^^^^^^^^^^^^^^^^^^^^

- **WARNING:** Conda lammps installation does NOT include ACE descriptor set, SPIN package, or new 
  LAMMPS settings needed for fitting neural networks. If you want to use these newer settings, 
  please build LAMMPS from source as explained in the `LAMMPS Installation`_ docs.

A minimal working environment can be set up using the Python distribution package Anaconda (https://www.anaconda.com).

After installing Anaconda:

#. Add conda-forge to your Conda install, if not already added::
    
        conda config --add channels conda-forge

#. Create a new Conda environment::

        conda create -n fitsnap python=3.10
        conda activate fitsnap

#. Install dependencies::

        conda install lammps
        python -m pip install numpy scipy scikit-learn virtualenv psutil pandas tabulate mpi4py Cython

#. Clone the FitSNAP repository::

        git clone https://github.com/FitSNAP/FitSNAP.git 

#. Add the cloned repository path to your PYTHONPATH environment variable, e.g. in :code:`~/.bashrc` 
   or :code:`~/.bash_profile`::
        
        FITSNAP_DIR=\path\to\FitSNAP
        export PYTHONPATH=$FITSNAP_DIR:$LAMMPS_DIR/python:$PYTHONPATH

- **TIP:** Periodically use the command :code:`git pull` in the cloned directory for updates 


