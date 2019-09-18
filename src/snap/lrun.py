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

class LRunException(SNAPException):
    pass

_staleLogDelta = 3.0 # Consider lammps.log.snap to be stale if it was last
                        # modified more than this many seconds before the
                        # recorded LAMMPS start time. A delta this large
                        # may be unnecessary, but according to the docs for
                        # os.stat, "Note: ...On Windows systems using the FAT
                        # or FAT32 file systems, st_mtime has 2-second
                        # resolution". So, let's be safe.
_numOmit = 13
_nktv2p = 1.6021765e6
_MAX_SUMSQERRX = 1e-8

_NIMAGE = 1
_shifter = zeros(( (2*_NIMAGE+1)**3,3))
_ctr = 0
for ia in range(-_NIMAGE,_NIMAGE+1):
    for ib in range(-_NIMAGE,_NIMAGE+1):
        for ic in range(-_NIMAGE,_NIMAGE+1):
            _shifter[_ctr] = array((ia,ib,ic),dtype=float64)
            _ctr += 1

class LAMMPSResults(object):
    def __init__(self,virials=None,energy=None,forces=None,
            energyBispectrum=None,virialsBispectrum=None,
            forcesBispectrum=None,numEachType=(0,0)):
        self.virials = virials
        self.energy = energy
        self.forces = forces
        self.energyBispectrum = energyBispectrum
        self.virialsBispectrum = virialsBispectrum
        self.forcesBispectrum = forcesBispectrum
        self.numEachType = numEachType

class _LogIterator(object):
    def __init__(self,numCoeffs=0,trainingConfigs=None):
        self.trainingConfigs = trainingConfigs
        self.numConfigs = len(self.trainingConfigs)
        self.numCoeffs = numCoeffs
        self.numCols = (1+numCoeffs)*options.numTypes
        nformat = len("%d" % self.numConfigs)
        self.dumpFormat = options.dumpPath + os.sep + "dump_%%0%dd" % nformat
        self.dumpdbFormat = options.dumpPath + os.sep + "dump_db_%%0%dd" % nformat
        self.logFormat = options.dumpPath + os.sep + "log_%%0%dd" % nformat
        self.bIndex = 0 # internal counter for iterator.
        # Extract column names from the first log file
        log1 = self.logFormat % 1
        try:
            columnNames = pizza.get_column_names(log1)
        except StandardError as e:
            raise LRunException("Error: Reported from pizza.py: " + \
                    "%s" % e.args[0])
        self.betaLabels = columnNames[_numOmit:]

    def __iter__(self):
        return self

    def extract_energy_betas(self,entries):
        eBetas = zeros(self.numCols)
       # verify dimensions of entries.
#        if len(entries) != 7*options.numTypes*self.numCoeffs:
#            raise LRunException("Error: Incorrect number of computes in " + \
#                    "LAMMPS logfile for energies in configuration %d." % self.bIndex)
        for itype in range(options.numTypes):
            e0 = itype*self.numCoeffs # start extraction here
            i0 = e0 + itype+1 # start insertion here (skip a column)
            e1 = e0+self.numCoeffs # finish extraction here
            i1 = i0 + self.numCoeffs # finish insertion here
#            print i0, i1, e0, e1, eBetas[0]
            eBetas[i0:i1] = entries[e0:e1]
        return eBetas

    def extract_virial_betas(self,entries):
        vBetas = zeros((6,self.numCols))
        # verify dimensions of entries.
#        if len(entries) != 7*options.numTypes*self.numCoeffs:
#            raise LRunException("Error: Incorrect number of computes in " + \
#                    "LAMMPS logfile for virials in configuration %d." % self.bIndex)
        for idim in range(6):
            for itype in range(options.numTypes):
                e0 = options.numTypes*self.numCoeffs+idim*self.numCoeffs+ \
                        itype*6*self.numCoeffs
                i0 = itype*self.numCoeffs + itype + 1
                e1 = e0 + self.numCoeffs
                i1 = i0 + self.numCoeffs
                vBetas[idim][i0:i1] = entries[e0:e1]
        return vBetas

    def extract_force_betas(self,raw):
        nAtoms = raw.shape[0]
        fBetas = zeros((3*nAtoms,self.numCols))
        for i, entries in enumerate(raw):
            for idim in range(3):
                for itype in range(options.numTypes):
                    e0 = idim*self.numCoeffs+itype*3*self.numCoeffs
                    i0 = self.numCoeffs*itype + itype + 1
                    e1 = e0 + self.numCoeffs
                    i1 = i0 + self.numCoeffs
                    ir = 3*i + idim # insertion row
                    fBetas[ir][i0:i1] = entries[e0:e1]
        return fBetas

    def delxsq(self,x1,x2,proj):
        delx0 = x1 - x2
        delx = delx0 + proj
        delxsq = (delx**2.0).sum(axis=1)
        return delxsq.min()

    def compute_cell_avsqerrx(self,cell,positions):
        tConfig = self.trainingConfigs[self.bIndex]
        proj = dot(tConfig.cell,_shifter.transpose()).transpose()
        delx = tConfig.positions - positions
        sumsqerrx = 0.0
        for t, l in zip(tConfig.positions, positions):
            sqerrx = self.delxsq(t,l,proj)
            sumsqerrx += sqerrx
        return sumsqerrx

    def accumulate_sumrijsq(self,cell,positions):
    #Compute sumrijsq for lammps results TODO: talk to Aidan about this
        proj = dot(cell,_shifter.transpose()).transpose()
        nAtoms = positions.shape[0]
        for i in range(nAtoms):
            for xj in positions[:i]:
                rijsqraw = self.delxsq(positions[i],xj,proj)
                self.sumrijsq += rijsqraw

    def next(self):
        lammpsIndex = self.trainingConfigs[self.bIndex].lammpsIndex
        ## Process the LAMMPS log file to get energy, volume, virials, and
        ## betas for energy and virials
        logName = self.logFormat % lammpsIndex
        try:
            energy, volume, virials, betas = \
                    pizza.process_log(logName,self.betaLabels)
        except StandardError as e:
            raise LRunException("Error: While processing " + logName + \
                    "Pizza.py reported: " + e.args[0])
        ## Process the LAMMPS dump file to get the cell, atomic positions and
        ## types, and force components
        dumpName = self.dumpFormat % lammpsIndex
        try:
            cell,positions, types, forces = pizza.process_dump(dumpName)
        except StandardError as e:
            raise LRunException("Error: While processing " + dumpName + \
                    "Pizza.py reported: " + e.args[0])
        ## Process the LAMMPS dump_db file to get force betas
        dumpdbName = self.dumpdbFormat % lammpsIndex
        try:
            forceBetas = pizza.process_dump_db(dumpdbName)
        except StandardError as e:
            raise LRunException("Error: While processing " + dumpdbName + \
                    "Pizza.py reported: " + e.args[0])
        # Extract and reorder betas parsed out of the log and dump files
        energyBetas = self.extract_energy_betas(betas)
        virialBetas = self.extract_virial_betas(betas)
        forceBetas = self.extract_force_betas(forceBetas)
        virialBetas *= _nktv2p/volume
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
        # compute cell avsqerrx
        if options.verifyConfigs:
            sumsqerrx = self.compute_cell_avsqerrx(cell,positions)
            nAtoms = positions.shape[0]
            if sumsqerrx/nAtoms > _MAX_SUMSQERRX:
                raise LRunException("Error: Mismatch in the atomic positions " + \
                        "for configuration %d between training " % self.bIndex + \
                        "set and LAMMPS results!")

        self.bIndex += 1 # advance bIndex for the next iteration
        return LAMMPSResults(virials=virials,energy=energy,forces=forces,
            energyBispectrum=energyBetas,virialsBispectrum=virialBetas,
            forcesBispectrum=forceBetas,numEachType=numEachType)

class LAMMPSInterface(object):
    def __init__(self, numCoeffs=0, trainingConfigs=None,libMode=False):
        self.numCoeffs = numCoeffs
        self.trainingConfigs=trainingConfigs
        self.numConfigs = len(self.trainingConfigs)
        self.libMode = libMode
        if options.runLammps:
            if not libMode:
                self._run_lammps_executable()
        # else: libMode, someday?

    def _run_lammps_executable(self):
        # determine concurrency to actually use.
        if options.maxConcurrency == 0: # useConcurrency from num. of cores
            useConcurrency = multiprocessing.cpu_count()
        elif options.maxConcurrency == 1: # serial
            useConcurrency = 1
        else:
            useConcurrency = options.maxConcurrency
        if useConcurrency > self.numConfigs: # disallowed by LAMMPS
            useConcurrency = self.numConfigs
        # construct argument list for subprocess.Popen
        args = []
        if useConcurrency != 1:
            args += shlex.split(options.mpiLauncherLAMMPS) + \
                    ["-n",str(useConcurrency)]
        args += shlex.split(options.lammpsPath) + ["-in", "in.snap",
                "-partition","%dx1" % useConcurrency,
                "-log","none",
                "-plog","none",
                "-pscreen","none",
                "-var","nfiles",str(self.numConfigs),
                "-var","DumpPath",str(options.dumpPath),
                "-var","DataPath",str(options.dataPath)]
        print "Launching LAMMPS."
#        print "Invocation Command: " + " ".join(args)
        LAMMPSstdout = open("out.out.snap","w")
        try:
            LAMMPSExec = subprocess.Popen(args,stdout=LAMMPSstdout,
                    stderr=subprocess.STDOUT)
        except OSError:
            raise LRunException("Error: Failed to launch LAMMPS with " + \
                    "command:\n%s" % " ".join(args))
        lammpsStartTime = time.time()
        print "Waiting.."
        returnCode = LAMMPSExec.wait()
        if returnCode != 0:
            raise LRunException("Error: LAMMPS invocation resulted in " + \
                    "a non-zero returncode.\nInvocation command: " + \
                    " ".join(args))
        LAMMPSstdout.close()
        print "Done."
        if options.staleOutputCheck:
            print "Checking for stale LAMMPS output."
            oldestAllowed = lammpsStartTime - _staleLogDelta
            try:
                artifacts = os.listdir(options.dumpPath)
            except:
                raise LRunException("Error: Attempt to examine LAMMPS " + \
                        "output in ./DumpSnap failed.")
            for artifact in artifacts:
                logStat = os.stat(options.dumpPath + os.sep + artifact)
                if logStat.st_mtime < oldestAllowed:
                    raise LRunException("Error: LAMMPS appeared to run " + \
                            "successfully, but log.lammps.snap is stale.")

    def __iter__(self):
        if not self.libMode:
            return _LogIterator(numCoeffs=self.numCoeffs,
                    trainingConfigs=self.trainingConfigs)
