Usage
=====

FitSNAP installation
--------------------

Simple conda install
^^^^^^^^^^^^^^^^^^^^
A minimal working environment can be set up using the Python distribution package Anaconda (https://www.anaconda.com).

After installing Anaconda:

#. Clone the FitSNAP repository::

        git clone https://github.com/FitSNAP/FitSNAP.git 

#. Add the cloned repository path to your PYTHONPATH environment variable::
        
        export PYTHONPATH=$FITSNAP_DIR:$LAMMPS_DIR/python:$PYTHONPATH

#. Add conda-forge to your Conda install, if not already added::
    
        conda config --add channels conda-forge

#. Create a new Conda environment::

        conda create -n fitsnap python=3.9
        conda activate fitsnap

#. Install packages::

        conda install lammps psutil pandas tabulate sphinx sphinx_rtd_theme mpi4py

- WARNING: Conda lammps installation does NOT include ACE descriptor set or SPIN package needed for these corresponding examples.

- TIP: Periodically use the command :code:`git pull` in the cloned directory for updates 


Install FitSNAP with latest (stable or development) LAMMPS version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Important: the following steps assume you are on a Unix (most likely Linux) system, using a command line, and use the command line program 'bash.' 

        - MacOS: depending on your OS version and hardware, you may need to change some of these steps (for example, your Terminal program may default to zsh instead of bash,  and LD_LIBRARY_PATH may instead be DYLD_LIBRARY_PATH and so on)
        - Windows: though FitSNAP should work fine in Windows, we are not able to provide instructions for Windows LAMMPS/FitSNAP installations at this time

Note: Both FitSNAP and LAMMPS have been optimized to work with MPI. For optimal performance of both, we recommend building and configuring your favored flavor of MPI before continuing (see 'Notes on building...' section below).

After setting up MPI (or not) and downloading/cloning LAMMPS:

#. Build a stable or development version of LAMMPS (see 'Notes on building...' section below)

#. After a successful LAMMPS build, to get FitSNAP to see the shared LAMMPS library, update your .bashrc or .bash_profile with the following: 

        - create a descriptive variable that points to your main LAMMPS directory
        - adjust your LD_LIBRARY_PATH variable to point to your lammps/build subdirecotry 
        - adjust your PYTHONPATH variable to point to your lammps/python subdirectory 
        - Example::
                
                # in e.g., ~/.bashrc
                LAMMPS_DIR=/your/path/to/your/lammps_version 
                export LD_LIBRARY_PATH=$LAMMPS_DIR/build:$LD_LIBRARY_PATH
                export PYTHONPATH=$LAMMPS_DIR/python:$PYTHONPATH

        - To confirm that these are working, fire up interactive python and try the following commands::

                import lammps
                lmp = lammps.lammps()

        -  If you see something like the following, you're good to go::

                Python 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:58:50) 
                [GCC 10.3.0] on linux
                Type "help", "copyright", "credits" or "license" for more information.
                >>> import lammps
                >>> lmp = lammps.lammps()
                LAMMPS (15 Sep 2022)
                >>> 

#. Clone the FitSNAP repository::

        git clone https://github.com/FitSNAP/FitSNAP.git 

#. Add the cloned repository path to your PYTHONPATH environment variable - don't forget about your previous LAMMPS variable!::
        
        export PYTHONPATH=$FITSNAP_DIR:$LAMMPS_DIR/python:$PYTHONPATH

#. Install the following required Python packages/libraries using conda, pip, or your favorite Python package management method. Example with conda::
        
        conda install psutil pandas scipy tabulate sphinx sphinx_rtd_theme

#. If MPI is configured on your system, we recommend installing mpi4py for optimal FitSNAP performance

        - If installing with a Python package manager, we strongly recommend using pip over conda as pip will auto-configure your mpi4py package to your system's defaut MPI version (usually what you used to build LAMMPS)


Notes on building MPI/LAMMPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**MPI for LAMMPS and FitSNAP**

Both LAMMPS and FitSNAP are parallelized for optimal performance. Though optional, using MPI to run both applications will very significantly speed up runtimes. 

We built MPI fromm source for LAMMPS and FitSNAP using OpenMPI version 4.1.4 (https://www.open-mpi.org/) and the instructions on that webiste (https://www.open-mpi.org/faq/?category=building#easy-build).

After building, add your openmpi executable path to your "PATH" variable as well so that LAMMPS can automatically find your MPI install, e.g.::
                
        # in e.g., ~/.bashrc
        MPI_DIR=/usr/local/openmpi     
        export PATH=$MPI_DIR/bin:$PATH

**LAMMPS for FitSNAP**

#. Clone the LAMMPS 'develop' or 'stable' branch::

        git clone https://github.com/lammps/lammps.git

        - TIP: Periodically use the command :code:`git pull` in the cloned directory for updates 

#. For the curses cmake (ccmake) method, build LAMMPS using the following steps:

        - In the main LAMMPS directory, create a new directory called 'build'
        - Cd into 'build' and use command 'ccmake ../cmake' (note double c in first 'ccmake'! that's for 'curses cmake' - curses is a simple GUI for command line)
        - Hit 'c' to set up the initial configuration, and toggle the following to TRUE: BUILD_MPI, BUILD_SHARED_LIBS,LAMMPS_EXCEPTIONS,ML_SNAP,(whatever other packages you want)
        - Hit 'c' again, check out (new) options, toggle what looks nice
        - Hit 'c' ad nauseam and check out ad nauseam
        - If nothing else changes and you see the 'generate' option appear, hit 'g' and exit screen
        - At the command line, type 'cmake --build . -jN' where N is the number of processors you can run simultaneously. N=8 or N=16 are good general settings, the more the faster
        - If all goes to plan, you should now have a usable LAMMPS executable 'lmp' in your build directory! If not, check out the cmake output for compile errors
        - To test your LAMMPS executable, attempt to run it with `./lmp`. If you load the LAMMPS command line, you're in business (CTRL + C to exit).

