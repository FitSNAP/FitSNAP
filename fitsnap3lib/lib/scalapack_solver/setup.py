from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# This line only needed if building with NumPy in Cython file.
from numpy import get_include
import subprocess
import os
from os import system
import warnings
import mpi4py

# fortran_mod_comp = 'mpif90 /usr/local/scalapack-2.0.2/SRC/pdgetri.f -c -o pdgetri_.o -O3 -fPIC'
# print(fortran_mod_comp)
# system(fortran_mod_comp)
#
# fortran_mod_comp = 'mpif90 /usr/local/scalapack-2.0.2/SRC/pdgetrf.f -c -o pdgetrf_.o -O3 -fPIC'
# print(fortran_mod_comp)
# system(fortran_mod_comp)
#
# shared_obj_comp = 'mpif90 scalapack_wrap.f90 -c -o scalapack_wrap.o -O3 -fPIC'
# print(shared_obj_comp)
# system(shared_obj_comp)


def runcommand(cmd):
    process = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    c = process.communicate()
    if process.returncode != 0:
        raise Exception("Something went wrong whilst running the command: %s" % cmd)
    return c[0]


def whichscalapack():
    # Figure out which Scalapack to use
    if 'MKLROOT' in os.environ:
        return 'intelmkl'
    else:
        return 'netlib'


def whichmpi():
    # Figure out which MPI environment this is
    import re
    try:
        # if scalapackversion == 'intelmkl':
        #     return 'mpich'
        mpiv = runcommand('mpirun -V')
        if re.search('Intel', mpiv):
            return 'intelmpi'
        elif re.search('Open MPI', mpiv):
            return 'openmpi'
    except:
        return 'mpich'
    warnings.warn('Unknown MPI environment.')
    return None


scalapackversion = whichscalapack()
mpiversion = whichmpi()

if mpiversion == 'openmpi':
    # Fetch the arguments for linking and compiling.
    mpilinkargs = runcommand('mpicc -showme:link').split()
    mpicompileargs = runcommand('mpicc -showme:compile').split()

if scalapackversion == 'intelmkl':
    # Set library includes taking into account which MPI library we are using.
    # Includes for specific machines are suggested by the Intel MKL link advisor online tool.
    # For example if using Intel compilers instead of GNU, one might switch
    # `mkl_gnu_thread` to `mkl_intel_thread` and `gomp` to `iomp5`.
    scl_lib = ['mkl_scalapack_ilp64', 'mkl_intel_ilp64', 'mkl_gnu_thread', 'mkl_core',
               'mkl_blacs_'+mpiversion+'_ilp64', 'gomp', 'pthread', 'mkl_avx512', 'mkl_def',
               'm', 'dl']
    scl_incl = os.environ['MKLROOT']+'/include'
    # The library directory also comes from the Intel MKL link advisor online tool.
    scl_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif scalapackversion == 'netlib':
    scl_lib = ['scalapack', 'gfortran']
    scl_libdir = [ os.path.dirname(runcommand('gfortran -print-file-name=libgfortran.a')) ]
else:
    raise Exception("Scalapack distribution unsupported. Please modify setup.py manually.")


ext_modules = [Extension(# module name:
                         'scalapack_funcs',
                         # source file:
                         sources=['scalapack.pyx'],
                         # Needed if building with NumPy. This includes the NumPy headers when compiling.
                         include_dirs=[get_include(), mpi4py.get_include(), scl_incl],
                         #include libraries 
                         library_dirs=scl_libdir, libraries=scl_lib,
                         # other compile args for gcc
                         extra_compile_args=["-DMKL_ILP64", "-m64"],
                         # other files to link to
                         extra_link_args=["-lm", "-ldl"] + mpilinkargs)]

setup (
          cmdclass = {'build_ext': build_ext},
          ext_modules = ext_modules
      )
