## FitSnap3 ScaLAPACK library

This library will generate a shared object library for multinode solving capabilities.
Files in this directory are written by Charlie A. Sievers

#### Building this library:

To build this library, use the following commands in this directory:

`python setup.py build_ext --inplace`
`python setup.py install`

#### Files in this Directory

setup.py 

Python file containing the setup configuration. (Cython setuptools)

scalapack.h

C header file containing ScaLAPACK function declarations.

scalapack.pxy

ScaLAPACK cython file.

Scalapack.pxd

Cython header file containing ScaLAPACK function declarations.

scalapack.py

Python file containing ScaLAPACK lstsq solver.

#### Important installation information

Requires mpi4py python package
Requires openmp and scalapack libraries
