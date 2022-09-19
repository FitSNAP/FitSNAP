<!----------------BEGIN-HEADER------------------------------------>
## FitSNAP3
A Python package for training machine learned potentials in the LAMMPS molecular dynamics package.

**Documentation page:** [https://fitsnap.github.io](https://fitsnap.github.io)

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### Original author:
    Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
    http://www.cs.sandia.gov/~athomps

#### Key contributors (alphabetical):
    Mary Alice Cusentino (Sandia National Labs)
    Nicholas Lubbers (Los Alamos National Lab)
    Drew Rohskopf (Sandia National Labs)
    Charles Sievers (UC Davis, Sandia National Labs)
    Adam Stephens (Sandia National Labs)
    Mitchell Wood (Sandia National Labs)

#### Additional authors (alphabetical):
    Elizabeth Decolvenaere (D. E. Shaw Research)
    Stan Moore (Sandia National Labs)
    Steve Plimpton (Sandia National Labs)
    Gary Saavedra (Sandia National Labs)
    Peter Schultz (Sandia National Labs)
    Laura Swiler (Sandia National Labs)

<!-----------------END-HEADER------------------------------------->

#### Using this package:
* [Required] This package expects a python 3.6+ version. Python dependencies: pandas scipy sphinx sphinx_rtd_theme psutil tabulate
* [Required] Compile LAMMPS (lammps.sandia.gov) as a shared library. Detailed instructions can be found in `docs/Installation.rst --> LAMMPS installation --> LAMMPS for FitSNAP` section of the manual. If you can open python and run `import lammps; lmp = lammps.lammps()` and it succeeds, you should be good.
* [Optional] To use neural network fitting functionality, install the Python package pytorch 
* [Optional] For optimal performance, also install your favored flavor of MPI (OpenMPI, MPICCH) and the Python package mpi4py. Note: if installing mpi4py with a Python package manager, we strongly recommend using pip over conda as pip will auto-configure your package to your system's defaut MPI version (usually what you used to build LAMMPS)
* [Optional] (Required for atomic cluster expansion, ACE, capabilities ) Along with compiling LAMMPS with all of the typical FitSNAP flags, turn the ML-PACE package on.
    * Clone the ML-PACE package with the implemented ACE descriptor computes into your build directory from: git@github.com:jmgoff/lammps-user-pace.git
    * Follow the README.md and INSTALL.md in this repo to build lammps with ACE descriptor calculators

#### Quick install (minimal working environment) using Conda:
(Similar instructions can be found in the manual under `docs/Installation.rst --> FitSNAP Installation --> Minimal conda install`)
* Clone this repository
* Add the cloned repository path to your PYTHONPATH environment variable (periodically `git pull` for code updates)
* Add conda-forge to your Conda install, if not already added \
    `conda config --add channels conda-forge` 
* Create a new Conda environment\
    `conda create -n fitsnap python=3.9; conda activate fitsnap;`
* Install the following packages:\
    `conda install lammps psutil pandas tabulate sphinx sphinx_rtd_theme mpi4py`
* WARNING: Conda lammps installation does NOT include ACE descriptor set or SPIN package needed for the corresponding examples.

#### Running:
* `(mpirun -np #) python -m fitsnap3 [options] infile` (optional)
* Command line options can be seen with `python -m fitsnap3 -h`
* Examples of published SNAP interatomic potentials can be found in `examples/`
