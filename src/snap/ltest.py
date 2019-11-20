# ---------------------------BEGIN-HEADER------------------------------------
# Copyright (2016) Sandia Corporation. 
# Under the terms of Contract DE-AC04-94AL85000 
# with Sandia Corporation, the U.S. Government 
# retains certain rights in this software. This 
# software is distributed under the GNU General 
# Public License.

# FitSNAP.py - A Python framework for fitting SNAP interatomic potentials

# Original author: Aidan P. Thompson, athomps@sandia.gov
# http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
# Key contributors:
# Mary Alice Cusentino
# Adam Stephens
# Mitchell Wood

# Additional authors: 
# Elizabeth Decolvenaere
# Stan Moore
# Steve Plimpton
# Gary Saavedra
# Peter Schultz
# Laura Swiler

# ----------------------------END-HEADER-------------------------------------

import subprocess
import shlex
import re
import time
import os
import multiprocessing
from numpy import *
from clopts import options
import training
import pizza
from snapexception import SNAPException

class LTestException(SNAPException):
    pass

class LAMMPSTestResults(object):
    def __init__(self,virials=None,energy=None,forces=None):
        self.virials = virials
        self.energy = energy
        self.forces = forces

class _LogTestIterator(object):
    def __init__(self,testingConfigs=None):
        self.testingConfigs = testingConfigs
        self.numConfigsTest = len(self.testingConfigs)
        nformat = len("%d" % self.numConfigsTest)
        self.dumpFormat = options.dumpPathTest + os.sep + "dump_%%0%dd" % nformat
        self.logFormat = options.dumpPathTest + os.sep + "log_%%0%dd" % nformat
        self.bIndex = 0 # internal counter for iterator.
        # Extract column names from the first log file
        log1 = self.logFormat % 1
        try:
            columnNames = pizza.get_column_names(log1)
        except StandardError as e:
            raise LTestException("Error: Reported from pizza.py: " + \
                    "%s" % e.args[0])

    def __iter__(self):
        return self

    def next(self):
        lammpsIndex = self.testingConfigs[self.bIndex].lammpsIndex
        ## Process the LAMMPS log file to get energy, volume, virials
        logName = self.logFormat % lammpsIndex
        try:
            energy, virials = \
                    pizza.process_log_test(logName)
        except StandardError as e:
            raise LTestException("Error: While processing " + logName + \
                    " Pizza.py reported: " + e.args[0])
        ## Process the LAMMPS dump file to get the cell, atomic positions and
        ## types, and force components
        dumpName = self.dumpFormat % lammpsIndex
        try:
            types, forces = pizza.process_dump_test(dumpName)
        except StandardError as e:
            raise LTestException("Error: While processing " + dumpName + \
                    " Pizza.py reported: " + e.args[0])

        self.bIndex += 1 # advance bIndex for the next iteration
        if options.numTypes == 2:
            numEachType = (types.count(1), types.count(2))
            energy -= (float(types.count(1)) * options.Eshift1 + float(types.count(2)) * options.Eshift2)
        elif options.numTypes == 3:
            numEachType = (types.count(1), types.count(2), types.count(3))
            energy -= (float(types.count(1)) * options.Eshift1 + float(types.count(2)) * options.Eshift2 + float(types.count(3)) * options.Eshift3)
        elif options.numTypes == 4:
            numEachType = (types.count(1), types.count(2), types.count(3), types.count(4))
            energy -= (float(types.count(1)) * options.Eshift1 + float(types.count(2)) * options.Eshift2 + float(types.count(3)) * options.Eshift3 + float(types.count(4)) * options.Eshift4)
        else:
            numEachType = (types.count(1), 0)
            energy -= (float(types.count(1)) * options.Eshift1)

        return LAMMPSTestResults(virials=virials,energy=energy,forces=forces)

class LAMMPSTestInterface(object):
    def __init__(self, testingConfigs=None,libMode=False):
        self.testingConfigs=testingConfigs
        self.numConfigsTest = len(self.testingConfigs)
        self.libMode = libMode
        if not libMode:
            self._run_lammps_test_executable()
        # else: libMode, someday?

    def _run_lammps_test_executable(self):
        # determine concurrency to actually use.
        if options.maxConcurrency == 0: # useConcurrency from num. of cores
            useConcurrency = multiprocessing.cpu_count()
        elif options.maxConcurrency == 1: # serial
            useConcurrency = 1
        else:
            useConcurrency = options.maxConcurrency
        if useConcurrency > self.numConfigsTest: # disallowed by LAMMPS
            useConcurrency = self.numConfigsTest
        # construct argument list for subprocess.Popen
        lmpinput = open("in.snaptest",'w')
        print >>lmpinput,\
        """
shell mkdir ${DumpPathTest}
label loop
variable i uloop ${nfilesTest} pad
log ${DumpPathTest}/log_${i}
units           metal
atom_style      %s
atom_modify map array sort 0 2.0
box tilt large
read_data ${DataPathTest}/data.lammps_${i}
mass * 1.0e-20
include ${PotFileTest}
thermo 100
thermo_style    custom step pe lx ly lz yz xz xy pxx pyy pzz pyz pxz pxy
thermo_modify format float %s
dump mydump all custom 1000 ${DumpPathTest}/dump_${i} id type x y z fx fy fz
dump_modify mydump sort id format float %s

neighbor 1.0e-20 nsq
run             0
clear
next i
jump SELF loop
        """ % ((options.atomstyle).lower(),'%20.15g','%20.15g')
        lmpinput.close()
        modname = "pot_%s.mod" % options.potentialFileName
        args = []
        if useConcurrency != 1:
            args += shlex.split(options.mpiLauncherLAMMPS) + \
                    ["-n",str(useConcurrency)]
        args += shlex.split(options.lammpsPath) + ["-in", "in.snaptest",
                "-partition","%dx1" % useConcurrency,
                "-log","none",
                "-plog","none",
                "-pscreen","none",
                "-var","nfilesTest",str(self.numConfigsTest),
                "-var","DumpPathTest",str(options.dumpPathTest),
                "-var","DataPathTest",str(options.dataPath),
                "-var","PotFileTest",str(modname)]
        print "Launching LAMMPS."
#        print "Invocation Command: " + " ".join(args)
        LAMMPSstdout = open("out.out.snap.test","w")
        try:
            LAMMPSExec = subprocess.Popen(args,stdout=LAMMPSstdout,
                    stderr=subprocess.STDOUT)
        except OSError:
            raise LTestException("Error: Failed to launch LAMMPS with " + \
                    "command:\n%s" % " ".join(args))
        lammpsStartTime = time.time()
        print "Waiting.."
        returnCode = LAMMPSExec.wait()
        if returnCode != 0:
            raise LTestException("Error: LAMMPS invocation resulted in " + \
                    "a non-zero returncode.\nInvocation command: " + \
                    " ".join(args))
        LAMMPSstdout.close()
        print "Done."
    def __iter__(self):
        if not self.libMode:
            return _LogTestIterator(testingConfigs=self.testingConfigs)

def test_system(A,b_training,b_reference,SNAPCoeff,configList,configRows,quantityRows,virialfactor):
    print "Running LAMMPS test."
    lti = LAMMPSTestInterface(testingConfigs=configList)

    numRows = b_reference.shape[0]

    print "Calculating test results."
    testvalues = zeros(numRows)  # Energies, forces, virials from LAMMPS
    for ind, result in zip(configRows,lti):
        testvalues[ind.eRow] = result.energy
        testvalues[ind.fRows] = result.forces.flatten()
        testvalues[ind.vRows] = result.virials*virialfactor

    # Scale energy rows by number of atoms
    for ind, config in zip(configRows,configList):
        invNumAtoms = 1.0/config.nAtoms
        testvalues[ind.eRow] *= invNumAtoms

    print "Writing test errors."
    # Compute model predictions and errors
    model = sum(A*SNAPCoeff,axis=1)
    error = testvalues - b_reference - model

    for quantity, func in zip( ("energy","force","virial"),
            ('eRows','fRows','vRows') ):
        errfp = open("SNAP%s_ErrTest.dat"%quantity,"w")
        rows = getattr(quantityRows,func)
        errfp.write("# %s, %s, %s, %s\n" % ("Test","Ref","Model=A.x","Test-Ref-Model"))
        errfp.write("# JSON Units, unweighted, energy scaled by nAtoms\n")
        for (valt,valr,valm,err) in zip(testvalues[rows],b_reference[rows],model[rows],error[rows]):
            errfp.write("%12.8g %12.8g %12.8g %12.8g\n" % (valt,valr,valm,err))
        errfp.close()
