<!----------------BEGIN-HEADER------------------------------------>
## FitSNAP3
A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
##

#### Original author:
    Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
    http://www.cs.sandia.gov/~athomps

#### Key contributors (alphabetical):
    Mary Alice Cusentino (Sandia National Labs)
    Nicholas Lubbers (Los Alamos National Lab)
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
* [Required] This package expects a python 3.6+ version.
* [Required] Compile LAMMPS (lammps.sandia.gov) as a shared library, detailed instructions can be found here `docs/LAMMPS.md`. If you can open python and run `import lammps; lmp = lammps.lammps()` and it succeeds, you should be good.
* [Optional] (Required for atomic cluster expansion, ACE, capabilities ) Along with compiling LAMMPS with all of the typical FitSNAP flags, turn the ML-PACE package on.
    * Clone the ML-PACE package with the implemented ACE descriptor computes into your build directory from: git@github.com:jmgoff/lammps-user-pace.git
    * Follow the README.md and INSTALL.md in this repo to build lammps with ACE descriptor calculators

#### Installing:
* For the most recent bug fixes/code changes:
    * Clone this repository and add it to your PYTHONPATH environment variable. Periodically `git pull` for code updates.
* A minimal working environment can be achieved with the following;
   `conda create -n fitsnap python=3.9 ;
     conda activate fitsnap ;
     conda install lammps psutil pandas tabulate mpi4py`
* NOTE : Conda lammps installation does NOT include ACE descriptor set or SPIN package needed for these corresponding examples. If the conda install of lammps fails, run `conda config --add channels conda-forge` and retry.
* See docs/INSTALL.md for more information.

#### Running:
* `(mpirun -np #) python -m fitsnap3 [options] infile` (optional)
* Command line options can be seen with `python -m fitsnap3 -h`
* Input files are described by `docs/TEMPLATE.in` and `docs/GROUPLIST.template`
* Examples of published SNAP interatomic potentials can be found in `examples/`
