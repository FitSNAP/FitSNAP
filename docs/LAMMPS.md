Instructions for running FitSNAP with the LAMMPS library interface

QUICK FIX
------------

Prerequisites for LAMMPS built with 'make' method:
1. LAMMPS compiled as a shared library with exception handling
2. <lammpsdir> is the full path name for a directory
   containing dirs src and python
3. <fitsnapdir> is the full path name for directory
   containing dirs fitsnap3 and examples
Note: if LAMMPS was built with cmake method, change 'src' to 'build' and add
appropriate compile flags in step B below (for more instructions refer to the
LAMMPS manual part 3.1)

# create a new directory

mkdir mytestdir
cd mytestdir

# make LAMMPS and FitSNAP visible

ln -s <fitsnapdir>/fitsnap3 .
ln -s <lammpsdir>/src/lib.lammps.so .
ln -s <lammpsdir>/python/lammps.py .

# run FitSNAP and check output against original

python -m fitsnap3 <fitsnapdir>/examples/Ta_Linear_JCP2014/Ta-example.in
diff Ta_metrics.csv <fitsnapdir>/examples/Ta_Linear_JCP2014/19Nov19_Standard

STEP BY STEP INSTRUCTIONS
-------------------------

A. Define locations for LAMMPS and FitSNAP
------------------------------------------

Add one of these to your login file (.profile, .cshrc, etc.):

setenv LAMMPSDIR <lammpsdir>    # (t)csh
export LAMMPSDIR=<lammpsdir>    # (ba)sh

Add one of these to your login file (.profile, .cshrc, etc.):

setenv FITSNAPDIR <fitsnapdir>  # (t)csh
export FITSNAPDIR=<fitsnapdir>  # (ba)sh

B. Build LAMMPS as shared library
---------------------------------

cd $LAMMPSDIR/src
Choose a LAMMPS Makefile, called Makefile.<machine>, where
<machine> denotes a string such as "serial" or "mpi".
Makefile.<machine> must be located
somewhere inside directory $LAMMPSDIR/src/MAKE.
Edit or MAKE/../Makefile.<machine>, and add -DLAMMPS_EXCEPTIONS to
the end of LMP_INC, e.g.

   LMP_INC = -DLAMMPS_EXCEPTIONS

make clean-shared_<machine> (only necessary after editing Makefile.machine)
make yes-snap
make -j mode=shlib <machine>

Make LAMMPS shared library visible:

Add one of these lines to your login file (.profile, .cshrc, etc.):

setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$LAMMPSDIR/src      # (t)csh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$LAMMPSDIR/src      # (ba)sh
setenv DYLD_LIBRARY_PATH ${DYLD_LIBRARY_PATH}:$LAMMPSDIR/src  # (t)sh on Macs
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:$LAMMPSDIR/src  # (ba)sh on Macs

For more info see:

https://lammps.sandia.gov/doc/Python_shlib.html
https://lammps.sandia.gov/doc/Build_settings.html#exceptions

C. Make LAMMPS Python module visible:
-------------------------------------

Add one of these lines to your login file (.profile, .cshrc, etc.):

setenv PYTHONPATH ${PYTHONPATH}:$LAMMPSDIR/python      # (t)csh
export PYTHONPATH=${PYTHONPATH}:$LAMMPSDIR/python      # (ba)sh

For more info see:

https://lammps.sandia.gov/doc/Python_install.html

D. Making FitSNAP Visible
-------------------------

Add one of these lines to your login file (.profile, .cshrc, etc.):

setenv PYTHONPATH ${PYTHONPATH}:$FITSNAPDIR/python      # normal (t)csh
export PYTHONPATH=${PYTHONPATH}:$FITSNAPDIR/python      # normal (ba)sh

E. Running FitSNAP
------------------

mkdir mytestdir
cd mytestdir
python -m fitsnap3 $FITSNAP/examples/Ta_Linear_JCP2014/19Nov19_Standard
diff Ta_metrics.csv $FITSNAP/examples/Ta_Linear_JCP2014/19Nov19_Standard


F. Troubleshooting
------------------

Here is a list of common error conditions that can occur when
FitSNAP is not set up correctly. Each problem is identified
by an error message that appears prominently near the end of
the screen output.

1. ".../python: No module named fitsnap3"
Python is unable to find the directory called fitsnap3. It must
either appear in the current working directory or the
directory containing it must be added to the PYTHONPATH variable.
See "D. Making FitSNAP Visible."

2. ".../python: 'fitsnap3' is a package and cannot be directly executed"
This means that you are using a Python earlier than 3.6. You
need to use Python 3.6 or later

3. "ImportError: Could not find lammps module."
Python is unable to find lammps.py. It must
either appear in the current working directory or the
directory containing it must be added to the PYTHONPATH variable.
See "C. Make LAMMPS Python module visible."

4. "OSError: dlopen(liblammps.so, 10): image not found"
Python is unable to find the LAMMPS shared library,
which is a file called liblammps.so. It must either appear in the
current working directory or the directory containing it must
be added to the PYTHONPATH variable. Or you forgot to build it.
See "B. Build LAMMPS as shared library."

5. "AttributeError: 'lammps' object has no attribute 'lmp'"
Same as 4

6. "Exception: Fitting interrupted! LAMMPS not compiled with C++ exceptions handling enable"
This is a FitSNAP error message indicating that the LAMMPS shared library was not compiled
with C++ exceptions handling. You should recompile LAMMPS with the
-DLAMMPS_EXCEPTIONS compile option. See "B. Build LAMMPS as shared library."
Alternatively, you can forge ahead by invoking FitSNAP with the command line option
--lammps_noexceptions. The downside is that any LAMMPS error messages will not
be written to standard error output i.e. the screen. They may still appear
in the LAMMPS log file, if one exists, but this is system dependent.
Also, using Python multiprocessing (-j --jobs command-line option)
in combination with a LAMMPS shared library compiled without C++ exceptions handling
can lead to undefined behavior if LAMMPS terminates
with an error, including Python processes that fail to terminate.

7. "ERROR: Unrecognized compute style 'sna<p or /atom>' is part of the SNAP package which is not enabled in this LAMMPS binary."
This is a LAMMPS error message indicating that the LAMMPS shared library was not
compiled with the SNAP package installed. See "B. Build LAMMPS as shared library."
