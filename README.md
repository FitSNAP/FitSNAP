<!----------------BEGIN-HEADER------------------------------------>

<img width="300" alt="FitSNAP" src="https://github.com/rohskopf/FitSNAP/blob/master/FitSNAP.png">


[![FitSNAP Pytests](https://github.com/FitSNAP/FitSNAP/actions/workflows/pytests.yaml/badge.svg?branch=master)](https://github.com/FitSNAP/FitSNAP/actions/workflows/pytests.yaml)

A Python package for machine learning potentials with [LAMMPS](https://github.com/lammps/lammps).

**Documentation page:** [https://fitsnap.github.io](https://fitsnap.github.io)

**Colab Python notebook tutorial:** [https://colab.research.google.com/github/FitSNAP/FitSNAP/blob/master/tutorial.ipynb](https://colab.research.google.com/github/FitSNAP/FitSNAP/blob/master/tutorial.ipynb)

#### How to cite 
We are currently working on an overview of the code in its current state, stay tuned!

#### Dependencies:

* This package expects Python 3.6+ 
* Python dependencies: `numpy pandas scipy psutil tabulate`
* [Compile LAMMPS as a shared library with python support](https://docs.lammps.org/Python_head.html). 
If you can run `import lammps; lmp = lammps.lammps()` without errors in your Python interpreter, 
you're good to go!
* [Optional] To use neural network fitting functionality, install PyTorch with `pip install torch`. 
* [Optional] For optimal performance, also install your favorite flavor of MPI (OpenMPI, MPICH) and 
the Python package `mpi4py`. If installing mpi4py with a Python package manager, we recommend using 
pip over conda as pip will auto-configure your package to your system's defaut MPI version 
(usually what you used to build LAMMPS).
* [Optional] For atomic cluster expansion (ACE) capabilities, build LAMMPS with the ML-PACE package, 
along with the `compute_pace` files from [https://github.com/jmgoff/lammps-user-pace](https://github.com/jmgoff/lammps-user-pace)

#### Quick install (minimal working environment) using Conda:

WARNING: Conda LAMMPS installation does NOT include ACE, SPIN, or neural network functionality. See 
the docs for details on how to install the current LAMMPS which has these functionalities.

* Clone this repository to get a FitSNAP directory:\
    `git clone https://github.com/FitSNAP/FitSNAP`
* Add `path/to/FitSNAP` path to your `PYTHONPATH` environment variable.
* Add conda-forge to your Conda install, if not already added:\
    `conda config --add channels conda-forge` 
* Create a new Conda environment:\
    `conda create -n fitsnap python=3.9; conda activate fitsnap;`
* Install the following packages:\
    `conda install lammps psutil pandas tabulate sphinx sphinx_rtd_theme mpi4py`
* Periodically `git pull` for code updates

#### Running:

* `(mpirun -np #) python -m fitsnap3 [options] infile`
* Command line options can be seen with `python -m fitsnap3 -h`
* Examples of published SNAP interatomic potentials are found in `examples/`
* Examples of running FitSNAP via the library interface are found in `examples/library`

#### Contributing:

* See our [Programmer Guide](https://fitsnap.github.io/Executable.html) on how to add new features.
* Get Sphinx with `pip install sphinx sphinx_rtd_theme` for adding new documentation, and see `docs/README.md` 
for how to build docs for your features. 
* **Feel free to ask for help!**

#### About
* Mitchell Wood and Aidan Thompson co-lead development of FitSNAP since 2016.
* The FitSNAP Development Team is the set of all contributors to the FitSNAP project, including all subprojects.
* The core development of FitSNAP is performed at the Center for Computing Research (CCR), Sandia National Laboratories, Albuquerque, New Mexico, USA 
* The original prototype of FitSNAP was developed in 2012 under a CIS LDRD project.

_Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
