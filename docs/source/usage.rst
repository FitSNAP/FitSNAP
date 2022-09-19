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


Install with latest (stable or development) LAMMPS version:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Important: the following steps assume you are on a Unix (most likely Linux) system, using a command line, and use the command line program 'bash.' 
        - MacOS: depending on your OS version and hardware, you may need to change some of these steps (for example, your Terminal program may default to zsh instead of bash,  and LD_LIBRARY_PATH may insetad be DYLD_LIBRARY_PATH and so on)
        - Windows: though FitSNAP should work fine in Windows, we are not able to provide instructions for Windows LAMMPS/FitSNAP installations at this time

#. For optimal FitSNAP performance, build your favored flavor of MPI. To be sure it's compatible with mpi4py, we built MPI from source with the instructions in the Appendix in mpi4py's manual (https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi) 
        - Important: add your openmpi install prefix (e.g. /usr/local/openmpi) to your "PATH" variable as well so that LAMMPS can automatically find your MPI install

#. Clone the 'develop' or 'stable' branch from https://github.com/lammps/lammps.

#. For the curses cmake (ccmake) method, build LAMMPS using the following steps:
        - In the main LAMMPS directory, create a new directory called 'build'
        - dd into 'build' and use command 'ccmake ../cmake' (note double c in first 'ccmake'! that's for 'curses cmake' - curses is a simple GUI for command line)
        - Hit 'c' to set up the initial configuration, and toggle the following to TRUE: BUILD_MPI, BUILD_SHARED_LIBS,LAMMPS_EXCEPTIONS,ML_SNAP,(whatever other packages you want)
        - Hit 'c' again, check out (new) options, toggle what looks nice
        - Hit 'c' ad nauseam and check out ad nauseam
        - If nothing else changes and you see the 'generate' option appear, hit 'g' and exit screen
        - at command line, type 'cmake --build . -jN' where N is the number of processors you can run simultaneously (N=8 or N=16 are good general settings, the more the faster!)
        - If all goes to plan, you should now have a usable LAMMPS executable 'lmp' in your build directory! If not, check out the cmake output for compile errors
                - To test your LAMMPS executable, attempt to run it with `./lmp`. If you load the LAMMPS command line, you're in business (CTRL + C to exit).

#. After a successful LAMMPS build, to get FitSNAP to see the shared LAMMPS library, update your .bashrc or .bash_profile with the following: 
        - create a descriptive variable 'LAMMPS_DIR=/your/path/to/your/lammps_version' : 
        - adjust your PYTHONPATH variable: 'export PYTHONPATH=$LAMMPS_DIR/python:$PYTHONPATH'
        - adjust your LD_LIBRARY_PATH variable: 'export LD_LIBRARY_PATH=$LAMMPS_DIR/build:$LD_LIBRARY_PATH'
        - to confirm that these are working, fire up interactive python and try the following commands::

                import lammps
                lmp = lammps.lammps()
        - if the LAMMPS version name pops up, you're good to go! Otherwise, check your environment variables one more time

#. Clone the FitSNAP repository

#. Add the cloned repository path to your PYTHONPATH environment variable (don't forget about your previous LAMMPS variable!): 'export PYTHONPATH=$FITSNAP_DIR:$LAMMPS_DIR/python:$PYTHONPATH'

#. Install the following required Python packages/libraries: psutil pandas scipy tabulate sphinx sphinx_rtd_theme

#. For optimal FitSNAP performance, if MPI is configured on your systel, also install mpi4py
        - If installing mpi4py with a Python package manager, we strongly recommend using pip over conda as pip will auto-configure your package to your system's defaut MPI version (usually what you used to build LAMMPS)




