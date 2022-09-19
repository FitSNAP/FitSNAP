Usage
=====

Installation
------------

To use FitSNAP, first install it according to the following instructions.

Simple conda install:
^^^^^^^^^^^^^^^^^^^^^
A minimal working environment can be set up using the Python distribution package Anaconda (https://www.anaconda.com).

After installing Anaconda:

#. Clone the FitSNAP repository 
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


Install with latest (development) LAMMPS version:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. For optimal FitSNAP performance, build your favored flavor of MPI. To be sure it's compatible with mpi4py, we built MPI from source with the instructions in the Appendix in mpi4py's manual (https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi) 
        - Note: add your openmpi install prefix to your "PATH" variable as well so that LAMMPS can find your MPI install!

#. Clone the 'develop' branch from https://github.com/lammps/lammps.

#. ccmake method: Build LAMMPS using the following steps:
        - in the main directory, create a new directory called 'build'
        - cd into 'build' and use command 'ccmake ../cmake' (note double c in 'ccmake'! that's for 'curses cmake' - curses is a simple GUI for command line)
        - hit 'c' for configure, and toggle the following tp TRUE: BUILD_MPI, BUILD_SHARED_LIBS,LAMMPS_EXCEPTIONS,ML_SNAP,(whatever other packages you want)
        - hit 'c' again, check out (new) options, toggle what looks nice
        - hit 'c' ad nauseam and check out ad nauseam
        - if nothing is changing or you're bored/done, hit 'g' for 'generate' and exit screen
        - at command line, type 'cmake --build . -jN' where N is the number of processors you can run simultaneously (N=8 or N=16 are good general settings, the more the faster!)
        - if all goes to plan, you should now have a usable LAMMPS executable 'lmp' in your build directory! if not, check out the cmake output for compile errors

#. to get FitSNAP to see the shared LAMMPS library, do the following in your .bashrc or .bash_profile: 
        - create a descriptive variable 'LAMMPS_DIR=/your/path/to/your/lammps_version' : 
        - adjust your PYTHONPATH variable: 'export PYTHONPATH=$LAMMPS_DIR/python:$PYTHONPATH'
        - adjust your LD_LIBRARY_PATH variable: 'export LD_LIBRARY_PATH=$LAMMPS_DIR/build:$LD_LIBRARY_PATH'
        - to confirm that these are working, fire up interactive python and try the following commands::

                import lammps
                lmp = lammps.lammps()
        - if the LAMMPS version name pops up, you're good to go! otherwise, check your environment variables one more time

#. Clone the FitSNAP repository

#. Add the cloned repository path to your PYTHONPATH environment variable

#. 




