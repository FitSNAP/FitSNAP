## FitSNAP ScaLAPACK library

This library will generate a shared object library for multinode solving capabilities.
Files in this directory are written by Charlie A. Sievers

### Building this library:

This sublibrary of FitSNAP is compiled and therefore depends on specific machine and module
settings. ScaLAPACK libraries require Intel MKL and specific library include settings for 
different machines are found online at the 
[Intel MKL link advisor tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html).

Once you have the linking and include options for your specific setup, include these options 
in the `scl_lib` list located in `setup.py`. Then we are ready to build.  

To build this library, use the following commands in this directory:

    python setup.py build_ext --inplace
    python setup.py install

These commands will compile the shared object library and then move the build to your 
environment site packages, respectively.

### Files in this Directory

#### setup.py 

Python file containing the setup configuration. (Cython setuptools)

#### scalapack.h

C header file containing ScaLAPACK function declarations.

#### scalapack.pxy

ScaLAPACK cython file.

#### Scalapack.pxd

Cython header file containing ScaLAPACK function declarations.

#### scalapack.py

Python file containing ScaLAPACK lstsq solver.

### Important installation information

Requires mpi4py python package, OpenMP, and ScaLAPACK libraries (often by loading a MKL module).

This package has been tested using GNU compilers, OpenMPI, and Intel MKL modules. Please 
find specific library linking flags for your specific module versions using the Intel 
MKL link advisor tool.
