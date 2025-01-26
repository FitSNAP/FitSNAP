
Compilation and installation
============================

Compiling and running on a local computer
-----------------------------------------

.. include:: md/Compiling-and-running-on-a-local-computer.md
  :parser: myst_parser.sphinx_


.. _supercomputers:

Compiling INQ on Supercomputing Clusters
----------------------------------------

.. include:: md/Compiling-inq-on-Supercomputing-Clusters.md
  :parser: myst_parser.sphinx_


ALCF Polaris
^^^^^^^^^^^^

.. include:: md/Compilation-ALCF-Polaris.md
  :parser: myst_parser.sphinx_

LLNL Sierra/Lassen
^^^^^^^^^^^^^^^^^^

.. include:: md/Compilation-LLNL-Sierra-Lassen.md
  :parser: myst_parser.sphinx_

LLNL Tioga (AMD)
^^^^^^^^^^^^^^^^

.. include:: md/Compilation-LLNL-Tioga-(AMD-machine).md
  :parser: myst_parser.sphinx_

OLCF Frontier (AMD)
^^^^^^^^^^^^^^^^^^^

.. include:: md/Compilation-OLCF-Frontier-(AMD).md
  :parser: myst_parser.sphinx_

NERSC Perlmutter (NVIDIA)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: md/Compilation-NERSC-Perlmutter-(NVIDIA).md
  :parser: myst_parser.sphinx_

Quartz
^^^^^^

.. include:: md/Quartz.md
  :parser: myst_parser.sphinx_

Compiling Parallel (MPI) HDF5 for INQ
-------------------------------------

.. include:: md/Compiling-Parallel-(MPI)-HDF5-for-INQ.md
  :parser: myst_parser.sphinx_

.. _cmake:

CMake
-----

This is a list of cmake options that are relevant for INQ / INQ template

.. list-table:: `cmake` options
  :width: 90%
  :align: center
  :widths: 20 20 60
  :header-rows: 1

  * - option
    - values
    - description
  * - `--fresh`
    -
    - Run the cmake configuration from scratch, ignoring what's already in the directory. We recommend you to always add this flag when calling `cmake`, especially when trying different configuration options. (This is only available from CMake 3.24 onwards, if you have a previous version you can get the same result by removing `CMakeCache.txt` and `CMakeFiles`.
  * - `--install-prefix=`
    -
    - The directory where INQ will be installed after compilation. This must be set and a place where the user is allowed to write.
  * - `-DCMAKE_BUILD_TYPE=`
    - `Release` (default), `Debug`
    - This sets the type of compilation of INQ. `-DCMAKE_BUILD_TYPE=Release` compile with maximum optimization and without debugging code, this is the fastest option that should be used for production runs. `-DCMAKE_BUILD_TYPE=Debug` add extra checks to ensure the code is running correctly, but it makes the code run slower. By default INQ is compiled in 'Release' mode.
  * - `-DENABLE_CUDA=yes`
    -
    - Enable compilation with CUDA. If this is set, INQ will try to run on an Nvidia GPU. Disabled by default.
  * - `-DENABLE_NCCL=yes`
    -
    - (Experimental) Use the NVIDIA NCCL library for communication. Disabled by default.

Cmake is requirement to build INQ (even if hidden behind a `./configure` script). Cmake handles INQ's dependencies, some of which are pretty new (in particular CUDA 11).

In general Cmake must be newer than the libraries it handles. Therefore a recent version of Cmake (e.g. above 3.16) might be required. For reference Ubuntu 20.04 has cmake 3.16 and Fedora 31 has cmake 3.17.

If your system supports modules, you can 

.. code:: bash

  module avail cmake
  module load cmake 3.17


There might be no option other than installing your own (and in your userspace).

.. code:: bash

  wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.sh
  sh ./cmake-3.21.1-linux-x86_64.sh --prefix=$HOME --skip-license
  ~/bin/cmake --version

The newer version of cmake will be in ~/bin/cmake. If you use the `./configure` script then you have to specify the command used to invoke cmake::

  CMAKE_CMD=~/bin/cmake ../configure ...

For other versions of cmake see https://cmake.org/download/

BOOST
-----

Install BOOST libraries
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: md/Install-BOOST-libraries.md
  :parser: myst_parser.sphinx_

Install BOOST
^^^^^^^^^^^^^

.. include:: md/Boost-installation.md
  :parser: myst_parser.sphinx_














.. SKIPPED  - [[Table of relevant GPU models]]

