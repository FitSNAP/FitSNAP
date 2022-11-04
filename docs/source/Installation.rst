Installation
============

This page documents how to properly install LAMMPS and FitSNAP. First we begin with how to install 
LAMMPS specifically for FitSNAP. If you do not want to manually install LAMMPS, please see our 
`Minimal conda install`_ for quickly getting started with FitSNAP.

.. _LAMMPS Installation:

LAMMPS Installation
-------------------

Since LAMMPS is the backbone of FitSNAP, we begin with instructions on how to install LAMMPS 
specifically for using FitSNAP.

MPI for LAMMPS and FitSNAP.
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both LAMMPS and FitSNAP are parallelized for optimal performance. Though optional, using MPI to run 
both applications will very significantly speed up runtimes. 

We built MPI fromm source for LAMMPS and FitSNAP using OpenMPI version 4.1.4 (https://www.open-mpi.org/) 
and the instructions on that webiste (https://www.open-mpi.org/faq/?category=building#easy-build).

After building, add your openmpi executable path to your :code:`PATH`` variable as well so that 
LAMMPS can automatically find your MPI install, e.g.::
                
        # in e.g., ~/.bashrc
        MPI_DIR=/usr/local/openmpi     
        export PATH=$MPI_DIR/bin:$PATH

LAMMPS for FitSNAP.
^^^^^^^^^^^^^^^^^^^

**First activate your Python virtual environment or conda environment.** Install the necessary 
pre-requisites to build LAMMPS with python using pip or conda::

        python -m pip install virtualenv numpy Cython # mpi4py (optional)

Then clone the LAMMPS repo::

        git clone https://github.com/lammps/lammps.git

This creates a :code:`lammps` directory, which we go to to create a custom LAMMPS and PyLammps build 
specifically for FitSNAP::

        cd /path/to/lammps
        mkdir build-fitsnap
        cd build-fitsnap
        cmake ../cmake -DLAMMPS_EXCEPTIONS=yes -DBUILD_SHARED_LIBS=yes -DMLIAP_ENABLE_PYTHON=yes -DPKG_PYTHON=yes -DPKG_ML-SNAP=yes -DPKG_ML-IAP=yes -DPKG_ML-PACE=yes -DPKG_SPIN=yes
        make ### Builds a LAMMPS executable and shared library
        make install-python ### Installs PyLammps so you can use the LAMMPS library in Python

This will create a LAMMPS executable :code:`lmp`, which should be used to run MD using FitSNAP fits.
This will also create a PyLammps interface located in your Python :code:`site-packages/lammps` 
directory. Set the following environment variables so that your Python can find LAMMPS::

    LAMMPS_DIR=/path/to/lammps
    export LD_LIBRARY_PATH=$LAMMPS_DIR/build-fitsnap:$LD_LIBRARY_PATH # Use DYLD_LIBRARY_PATH for MacOS
    export PYTHONPATH=$LAMMPS_DIR/python:$PYTHONPATH

To make sure MPI is working, make a Python script called :code:`test.py` with the following::

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print("Proc %d out of %d procs" % (comm.Get_rank(),comm.Get_size()))

And see the output for each processor by running::

    # NOTE: the line order is not deterministic
    $ mpirun -np 4 python test.py
    Proc 0 out of 4 procs
    Proc 1 out of 4 procs
    Proc 2 out of 4 procs
    Proc 3 out of 4 procs

Also make sure your Python LAMMPS library is working by firing up your Python interpreter and doing::

    import lammps
    lmp = lammps.lammps()

which should produce no errors.

**Now you have LAMMPS and PyLammps ready to use FitSNAP!**

Alternatively, LAMMPS can be built with the GUI CMake curses interface as explained below. With the 
CMake curses (ccmake) GUI interface, build LAMMPS using the following steps:

  - In the main LAMMPS directory, create a new directory called :code:`build-fitsnap`
  - Cd into :code:`build-fitsnap` and do :code:`ccmake ../cmake`` (note double c in first 'ccmake'! 
    that's for 'curses cmake' - curses is a simple GUI for command line)
  - Hit 'c' to set up the initial configuration, and toggle the following to TRUE: BUILD_MPI, 
    BUILD_SHARED_LIBS,LAMMPS_EXCEPTIONS,ML_SNAP,(whatever other packages you want)
  - Hit 'c' again, check out (new) options, toggle what looks nice
  - Hit 'c' ad nauseam and check out ad nauseam
  - If nothing else changes and you see the 'generate' option appear, hit 'g' and exit screen
  - At the command line, type :code:`cmake --build . -jN`` where N is the number of processors you 
    can run simultaneously. N=8 or N=16 are good general settings, the more the faster
  - If all goes to plan, you should now have a usable LAMMPS executable 'lmp' in your :code:`build_fitsnap` 
    directory! If not, check out the cmake output for compile errors
  - To test your LAMMPS executable, attempt to run it with :code:`./lmp`. If you load the LAMMPS 
    command line, you're in business (CTRL + C to exit).

After completing this LAMMPS installation, please see `Install FitSNAP with latest LAMMPS`_ to use FitSNAP.


FitSNAP Installation
--------------------

There are two primary ways to get started with FitSNAP: (1) a simple conda environment using the 
packaged LAMMPS that ships with conda, and (2) building LAMMPS manually. The latter option allows 
for more recent LAMMPS features.

.. _Minimal conda install:

Minimal conda install
^^^^^^^^^^^^^^^^^^^^^
A minimal working environment can be set up using the Python distribution package Anaconda (https://www.anaconda.com).

After installing Anaconda:

#. Clone the FitSNAP repository::

        git clone https://github.com/FitSNAP/FitSNAP.git 

#. Add the cloned repository path to your PYTHONPATH environment variable, e.g. in :code:`~/.bashrc` 
   or :code:`~/.bash_profile`::
        
        FITSNAP_DIR=\path\to\FitSNAP
        export PYTHONPATH=$FITSNAP_DIR:$LAMMPS_DIR/python:$PYTHONPATH

#. Add conda-forge to your Conda install, if not already added::
    
        conda config --add channels conda-forge

#. Create a new Conda environment::

        conda create -n fitsnap python=3.9
        conda activate fitsnap

#. Install packages (pytorch is optional)::

        conda install lammps psutil pandas tabulate sphinx sphinx_rtd_theme mpi4py pytorch

- **WARNING:** Conda lammps installation does NOT include ACE descriptor set, SPIN package, or new 
  LAMMPS settings needed for fitting neural networks. If you want to use these newer settings, 
  please build LAMMPS from source as explained in the `LAMMPS Installation`_ docs.

- **TIP:** Periodically use the command :code:`git pull` in the cloned directory for updates 

.. _Install FitSNAP with latest LAMMPS:

Install FitSNAP with latest LAMMPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following numbered steps assume you are on a Linux system, using a :code:`bash` command line. 
Notes for other operating systems:

  - **MacOS:** depending on your OS version and hardware, you may need to change some of these 
    steps (for example, your Terminal program may default to zsh instead of bash, and 
    :code:`LD_LIBRARY_PATH` may instead be :code:`DYLD_LIBRARY_PATH` and so on)
  - **Windows:** though FitSNAP should work fine in Windows, we are not able to provide 
    instructions for Windows LAMMPS/FitSNAP installations at this time.

Both FitSNAP and LAMMPS have been optimized to work with MPI. For optimal performance of 
both, we recommend building and configuring your favored flavor of MPI before continuing 
(see `LAMMPS Installation`_ docs).

After setting up MPI (or not) and downloading/cloning LAMMPS:

#. Build a stable or development version of LAMMPS (see `LAMMPS Installation`_ docs)

#. After a successful LAMMPS build, to get FitSNAP to see the shared LAMMPS library, update your 
   :code:`~/.bashrc`` or :code:`~/.bash_profile`` with the following:

        - create a descriptive variable that points to your main LAMMPS directory
        - adjust your :code:`LD_LIBRARY_PATH`` variable to point to your :code:`lammps/build-fitsnap` 
          subdirecotry 
        - adjust your :code:`PYTHONPATH` variable to point to your :code:`lammps/python` subdirectory 
        - Example::
                
                # in e.g., ~/.bashrc
                LAMMPS_DIR=/path/to/lammps
                export LD_LIBRARY_PATH=$LAMMPS_DIR/build-fitsnap:$LD_LIBRARY_PATH
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

#. Add the cloned repository path to your PYTHONPATH environment variable::
        
        FITSNAP_DIR=\path\to\FitSNAP
        export PYTHONPATH=$FITSNAP_DIR:$PYTHONPATH

#. Install the following required Python packages/libraries using conda, pip, or your favorite 
   Python package management method. Example with conda::
        
        conda install psutil pandas scipy tabulate sphinx sphinx_rtd_theme

#. If MPI is configured on your system, we recommend installing mpi4py for optimal FitSNAP performance

        - If installing with a Python package manager, we strongly recommend using pip over conda 
          as pip will auto-configure your mpi4py package to your system's defaut MPI version 
          (usually what you used to build LAMMPS)

#. You should now be able to run the FitSNAP examples in :code:`FitSNAP/examples`.

.. _LAMMPS PACE install:

LAMMPS PACE install
^^^^^^^^^^^^^^^^^^^

After installing scipy and the sym_ACE library available at: https://github.com/jmgoff/sym_ACE_lite.git and for developers at https://github.com/jmgoff/sym_ACE.git

#. Clone the lammps repository set up a typical LAMMPS build the ML-PACE library enabled::

        git clone -b compute-pace git@github.com:jmgoff/lammps_compute_PACE.git
        cd lammps_compute_PACE
        mkdir build && cd build

# Set up a typical LAMMPS build the ML-PACE library enabled::

        cmake -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D BUILD_SHARED_LIBS=on -D CMAKE_BUILD_TYPE=Debug -D PKG_ML-IAP=on -D PKG_ML-PACE=on -D PKG_ML-SNAP=on -D BUILD_MPI=on -D BUILD_OMP=off  -D CMAKE_INSTALL_PREFIX=<$HOME>/.local -D ../cmake/

# Next, download the modified lammps-user-pace code that contains extra arrays for breaking out descriptor contributions::

        git clone https://github.com/jmgoff/lammps-user-pace-1
        cp lammps-user-pace-1/ML-PACE/ace-evaluator/ace_evaluator.* ./lammps-user-pace-v.2022.10.15/ML-PACE/ace-evaluator/
        make -j
        make install


# Now, set up paths::

        INSTALL_PATH=$CMAKE_INSTALL_PREFIX/.local
        export PYTHONPATH=$PYTHONPATH:$INSTALL_PATH/lib/python3.<version>/site-packages
        export PYTHONPATH=$PYTHONPATH:$INSTALL_PATH/lib
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_PATH/lib
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<path>/<to>/<lammps>/build

