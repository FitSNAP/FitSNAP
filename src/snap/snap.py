#!/usr/bin/env python

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

import pickle, os, numpy, sys
import clopts, training, lrun, ldeck, lrundeck, postfit, ltest
from clopts import options
from numpy import *
from scipy.linalg import lstsq
#from sklearn import linear_model, decomposition
from snapexception import SNAPException
import json
import time
import shlex
import subprocess
from snapexception import SNAPException

version = "22Mar18"
class TrainingException(SNAPException):
    pass

class LRunException(SNAPException):
    pass

def populate_Ab(b_training, configList, configRows, numRows,numCols,numCoeffs,virialfactor):
    # Populate A and b. If user requested that LAMMPS not be run, attempt to
    # read in serialized A and b. If this fails, or if user requested that
    # LAMMPS be run, call the LAMMPSInterface constructor and build A and b by
    # parsing LAMMPS results.
    if not options.runLammps:
        fullPath = options.dumpPath+os.sep+"Absystem.dat"
        foundAbPickle = True
        try:
            fp = open(fullPath,"r")
        except:
            foundAbPickle = False
        if foundAbPickle:
                args_pickle = []
                args_pickle += shlex.split(options.mpiLauncherLSTSQ) + ["python","snap/pickleMPI.py"] + shlex.split(fullPath)
                print('Attempting to run MPI pickle script')
                pickle_stdout = open("out.pickle","w")
                try:
                        pickleExec = subprocess.Popen(args_pickle,stdout=pickle_stdout,
                                stderr=subprocess.STDOUT)
                except OSError:
                        raise LRunException("Error: Failed to launch MPI pickle script with " + \
                             "command:\n%s" % " ".join(args_pickle))
                pickleStartTime = time.time()
                print "Waiting.."
                returnCode = pickleExec.wait()
                if returnCode != 0:
                        raise LRunException("Error: Pickle MPI script invocation resulted in " + \
                        "a non-zero returncode.\nInvocation command: " + \
                        " ".join(args_pickle))
                pickle_stdout.close()
                print "Done."
                if os.path.exists('A_pickle.out.npy'):
                        A = numpy.load('A_pickle.out.npy')
                        b = numpy.load('b_pickle.out.npy')
                        b_reference = numpy.load('b_reference_pickle.out.npy')
                else:
                        foundAbPickle = False

#            print "Previously generated A and b (Absystem.dat) found."
#            p = pickle.Unpickler(fp)
#            try:
#                A, b, b_reference = p.load()
#            except (ValueError, KeyError):
#                print "Warning: Attempt to import previously " + \
#                    "generated A and b from %s failed. Will now " % fullPath + \
#                    "attempt to parse existing LAMMPS results."
#                foundAbPickle = False
#            finally:
#                fp.close()
        else:
            print "Previously generated A and b (Absystem.dat) NOT found."
    if options.runLammps or (not options.runLammps and not foundAbPickle):
        # Allocate space for A and b.
        A = zeros((numRows,numCols))
        print "Number of Rows, Columns in A: ", int(numRows) , int(numCols)
        b_reference = zeros(numRows)  # Energies, forces, virials from LAMMPS
        # Run LAMMPS.
        li = lrun.LAMMPSInterface(numCoeffs=numCoeffs,trainingConfigs=configList)
        # insert LAMMPS results into A and b_reference
        print "Parsing all LAMMPS results."

        print "Constructing system of equations."
        for ind, result in zip(configRows,li):
            b_reference[ind.eRow] = result.energy
            A[ind.eRow] = result.energyBispectrum
            for itype in range(options.numTypes):
                A[ind.eRow][itype*(numCoeffs+1)] = result.numEachType[itype]
            b_reference[ind.fRows] = result.forces.flatten()
            A[ind.fRows] = result.forcesBispectrum
            b_reference[ind.vRows] = result.virials*virialfactor
            A[ind.vRows] = result.virialsBispectrum*virialfactor

        # Scale energy rows by number of atoms
        for ind, config in zip(configRows,configList):
            invNumAtoms = 1.0/config.nAtoms
            b_reference[ind.eRow] *= invNumAtoms
            b_training[ind.eRow] *= invNumAtoms
            A[ind.eRow] *= invNumAtoms

        b = b_training-b_reference

    return A, b, b_reference

def populate_HalfAb(b_training, configList, configRows, numRows,numCols,numCoeffs, virialfactor):
    if not options.runLammps:
#        except (ValueError, KeyError):
        print "Warning: the ability to read in an old Ab while preserving old beta terms is not currently possible, exiting."
        exit(0)
    fp = open(options.coeffold,"r")
    print "Previously generated beta terms found."
    with fp as ins:
            raw = []
            for line in ins:
                    raw.append(line)
	    if int(options.freezeold)==2:
		    del raw[numCoeffs+5]
	    elif int(options.freezeold)==3:
	            del raw[numCoeffs+5]
	            del raw[2*numCoeffs+5]

            beta_old = [float(x.strip()) for x in raw[4:]]
    fp.close()

    if options.runLammps or (not options.runLammps and not foundAbPickle):
        # Allocate space for A and b.
        A = zeros((numRows,numCols))
        print "Number of Rows, Columns in A: ", int(numRows) , int(numCols)
        A_1half = zeros((numRows,int(options.freezeold)*numCoeffs))
        A_2half = zeros((numRows,(numCols-int(options.freezeold)*numCoeffs)))
        b_reference = zeros(numRows)  # Energies, forces, virials from LAMMPS
        # Run LAMMPS.
        li = lrun.LAMMPSInterface(numCoeffs=numCoeffs,trainingConfigs=configList)
        # insert LAMMPS results into A and b_reference
        print "Parsing all LAMMPS results."

        print "Constructing system of equations just for the free element."
        for ind, result in zip(configRows,li):
            b_reference[ind.eRow] = result.energy
            A[ind.eRow] = result.energyBispectrum
            for itype in range(options.numTypes):
                A[ind.eRow][itype*(numCoeffs+1)] = result.numEachType[itype]
            b_reference[ind.fRows] = result.forces.flatten()
            A[ind.fRows] = result.forcesBispectrum
            b_reference[ind.vRows] = result.virials*virialfactor
            A[ind.vRows] = result.virialsBispectrum*virialfactor
        A_1half = A[:,:(int(options.freezeold)*(numCoeffs+1))]
        A_2half = A[:,(int(options.freezeold)*(numCoeffs+1)):]
        print "Shape of A_1: ", A_1half.shape
        colsum_A1b = 0.0
        b_old = zeros(numRows)
        for row in range(numRows):
                for col in range(int(options.freezeold)*(numCoeffs)):
                        colsum_A1b += A_1half[row,col]*beta_old[col]
        	b_old[row] += colsum_A1b
        	colsum_A1b = 0.0

        # Scale energy rows by number of atoms
        for ind, config in zip(configRows,configList):
            invNumAtoms = 1.0/config.nAtoms
            b_training[ind.eRow] *= invNumAtoms
            b_reference[ind.eRow] *= invNumAtoms
            b_old[ind.eRow] *= invNumAtoms
            A[ind.eRow] *= invNumAtoms
            A_1half[ind.eRow] *= invNumAtoms
            A_2half[ind.eRow] *= invNumAtoms

        b = b_training-b_reference-b_old

    return A, b, b_reference, A_1half, A_2half, beta_old

def calculate_SNAP():
    # Write the LAMMPS script to compute all bispectrum coefficients and
    # reference energies. The number of bis. coeff. is used in several
    # places.
    numCoeffs = ldeck.gen_lammps_script()
    coeffindices = ldeck._generatecoeffindices(options.twojmax) #returns blist(i,j1,j2,j)
    if options.quadratic == 1:
	numLinear = numCoeffs
	numCoeffs = (numCoeffs*numCoeffs + numCoeffs)/2 + numLinear
	print "Num Linear, Quad", numLinear, numCoeffs
    # number of coefficients per type,
    # Read in the training set, processing the JSON data if necessary.
    # configList is a list of Config objects. Each contains information about
    # a training set configuration. {config,group,quanity}Rows contain
    #   slices into the matrices (A and b) used to do the regression.
    #   b_training is a numpy array containing the Quest energies, forces, and
    #   virials.
    # not incl. constant
    configList, dataStyles, configRows, groupRows, quantityRows, b_training = training.read_training_set()
    numCols = (1+numCoeffs)*options.numTypes
    numRows = b_training.shape[0]
    # Read in group weights. configList is passed in to help validate the
    # contents
    groupWeights = training.read_group_weights(configList)
    # Set the values of JSON/LAMMPS conversion factors
    if dataStyles['StressStyle'] == 'bar':
        virialfactor = 1.0
    elif dataStyles['StressStyle'] == 'kbar' or dataStyles['StressStyle'] == 'kB':
        virialfactor = 0.001
    else:
        raise TrainingException("Error: Encountered unknown dataStyles['StressStyle'] " + \
                                "%s while calculating virialfactor " \
                                % dataStyles['StressStyle'])

    # Populate A and b. They may be read in or computed, depending on
    # user options.
    if int(options.freezeold) == 0:
            A, b, b_reference = populate_Ab(b_training, configList, configRows, numRows, numCols, numCoeffs, virialfactor)
    elif int(options.freezeold) >= 1:
            A, b, b_reference, A_1half, A_2half, beta_old  = populate_HalfAb(b_training, configList, configRows, numRows, numCols, numCoeffs, virialfactor)
    # Construct the weight vector
    w = zeros(numRows)
    for name, ind in groupRows.iteritems():
        w[ind.eRows] = groupWeights[name].eweight
        w[ind.fRows] = groupWeights[name].fweight
        w[ind.vRows] = groupWeights[name].vweight
    # Write out abtotal.dat (formatted for solution by llsw) and
    # Absystemd.dat, where A and b are serialized. These files should be
    # written now, before a constant column is (potentially) deleted
    # from A.
    # Remove column for the constant term if requested
    if options.numTypes == 1 and options.numConstants == 0 and int(options.freezeold) == 0:
	A[:,0] = 0
    elif options.numTypes == 2 and options.numConstants == 1 and int(options.freezeold) == 0:
	A[:,numCoeffs+1] = 0
    elif options.numTypes == 2 and options.numConstants == 0 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
    elif options.numTypes == 3 and options.numConstants == 2 and int(options.freezeold) == 0:
	A[:,0] = 0
    elif options.numTypes == 3 and options.numConstants == 1 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
    elif options.numTypes == 3 and options.numConstants == 0 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
	A[:,2*numCoeffs+2] = 0
    elif options.numTypes == 4 and options.numConstants == 3 and int(options.freezeold) == 0:
	A[:,0] = 0
    elif options.numTypes == 4 and options.numConstants == 2 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
    elif options.numTypes == 4 and options.numConstants == 1 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
	A[:,2*numCoeffs+2] = 0
    elif options.numTypes == 4 and options.numConstants == 0 and int(options.freezeold) == 0:
	A[:,0] = 0
	A[:,numCoeffs+1] = 0
	A[:,2*numCoeffs+2] = 0
	A[:,3*numCoeffs+3] = 0
    elif options.numTypes == 2 and options.numConstants == 0 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
    elif options.numTypes == 3 and options.numConstants == 0 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
	A_2half[:,numCoeffs+1] = 0
    elif options.numTypes == 3 and options.numConstants == 1 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
    elif options.numTypes == 4 and options.numConstants == 0 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
	A_2half[:,numCoeffs+1] = 0
	A_2half[:,2*numCoeffs+2] = 0
    elif options.numTypes == 4 and options.numConstants == 1 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
	A_2half[:,numCoeffs+1] = 0
    elif options.numTypes == 4 and options.numConstants == 2 and int(options.freezeold) == 1:
	A_2half[:,0] = 0
    else:
	print "No Column Reduction Needed for NumConst<NumTypes"
    # Modify with gamma
    #if options.gamma != 1.0:
    #    for ind in configRows:
    #        A[ind.fRows] = A[ind.fRows]*options.gamma*(A[ind.eRow]**(options.gamma-1.0))
    #        A[ind.eRow] = A[ind.eRow]**options.gamma

    # Solve system and write out result.
    print "Performing the fit using ",options.solver
    if int(options.freezeold) == 0:
        print "Shape of A, b: ", A.shape, b.shape
	if options.solver == "SVD":
	#	SNAPCoeff, res, rank, s = lstsq(A*w[:,newaxis],b*w)
                numpy.save('A.out',A)
                numpy.save('b.out',b)
                numpy.save('w.out',w)

                args_lstsq = []
                dirname, filename = os.path.split(os.path.abspath(__file__))
                args_lstsq += shlex.split(options.mpiLauncherLSTSQ) + ["python","%s/lstsqMPI.py" % "".join(dirname)]
                print('Attempting to run LSTSQ script')
                LSTSQstdout = open("out.lstsq","w")
                try:
                        LSTSQExec = subprocess.Popen(args_lstsq,stdout=LSTSQstdout,
                                stderr=subprocess.STDOUT)
                except OSError:
                        raise LRunException("Error: Failed to launch LSTSQ script with " + \
                             "command:\n%s" % " ".join(args))
                lstsqStartTime = time.time()
                print "Waiting.."
                returnCode = LSTSQExec.wait()
                if returnCode != 0:
                        raise LRunException("Error: LSTSQ script invocation resulted in " + \
                        "a non-zero returncode.\nInvocation command: " + \
                        " ".join(args_lstsq))
                LSTSQstdout.close()
                print "Done."

                SNAPCoeff = numpy.load('SNAPCoeff.out.npy')
                res = numpy.load('res.out.npy')
                s = numpy.load('s.out.npy')
                rankArray = []
                rankFile = open('rank.out','r')
                lines = rankFile.readlines()
                for i in range(0,len(lines)):
                        line = lines[i]
                        columns = line.split()
                        rankArray.append(columns[0])
                rank = rankArray[0]



	elif options.solver == "LASSO":
		reg = linear_model.Lasso (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6)
		reg.fit(A*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
	elif options.solver == "RIDGE":
		reg = linear_model.Ridge (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6)
		reg.fit(A*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
	elif options.solver == "ELASTIC":
		reg = linear_model.ElasticNet (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6,l1_ratio=float(options.normratio))
		reg.fit(A*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
        postfit.compute_mean_errors(A,b,SNAPCoeff,configList,configRows,groupRows,quantityRows)
        # Write out residuals, energies, etc, if requested
	if int(options.PCAsize) > 0:
		postfit.compute_pca(A,b,w,SNAPCoeff,configList,configRows,groupRows,quantityRows)
        if options.writeTrainingErrors:
            postfit.write_detailed_results(A,b_training,b_reference,SNAPCoeff,configList,configRows,quantityRows)
	print "Writing SNAP coefficients."
	fp = open("SNAPcoeff.dat","w")
	line=0
	for c in SNAPCoeff:
            fp.write("%16.12f # B%s \n"%(c, coeffindices[line]))
            if (line>=(len(coeffindices)-1)):
                line=0
            else:
                line+=1
        # Write SNAP potential for use in LAMMPS
        lrundeck.gen_lammps_script(SNAPCoeff,numCoeffs,coeffindices)
        # Run SNAP potential on training data and check consistency
        if options.computeTestingErrors:
            ltest.test_system(A,b_training,b_reference,SNAPCoeff,configList,configRows,quantityRows,virialfactor)

    elif int(options.freezeold) >= 1:
        print "Shape of A_2, b: ", A_2half.shape, b.shape
	if options.solver == "SVD":
		SNAPCoeff, res, rank, s = lstsq(A_2half*w[:,newaxis],b*w)
	elif options.solver == "LASSO":
		reg = linear_model.Lasso (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6)
		reg.fit(A_2half*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
	elif options.solver == "RIDGE":
		reg = linear_model.Ridge (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6)
		reg.fit(A_2half*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
	elif options.solver == "ELASTIC":
		reg = linear_model.ElasticNet (alpha=float(10**(float(options.normweight))), fit_intercept=False,max_iter=1E6,l1_ratio=float(options.normratio))
		reg.fit(A_2half*w[:,newaxis],b*w)
		SNAPCoeff=reg.coef_
        postfit.compute_mean_errors(A_2half,b,SNAPCoeff,configList,configRows,groupRows,quantityRows)
# Write out residuals, energies, etc, if requested
	if int(options.PCAsize) > 0:
		postfit.compute_pca(A_2half,b,w,SNAPCoeff,configList,configRows,groupRows,quantityRows)
        if options.writeTrainingErrors:
            postfit.write_detailed_results(A_2half,b_training,b_reference,SNAPCoeff,configList,configRows,quantityRows)
        print "Writing frozen SNAP coefficients + new values from current fit."
        fp = open("SNAPcoeff.dat","w")
        for c in beta_old:
               fp.write("%16.12f\n"%c)
        line=0
        for c in SNAPCoeff:
            fp.write("%16.12f # B%s \n"%(c, coeffindices[line]))
            line+=1
        fp = open("SNAPcoeff.dat","r")
        with fp as ins:
            SNAPCoeff = []
            for line in ins:
                    SNAPCoeff.append(line)
            SNAPCoeff = [float(x.strip()) for x in SNAPCoeff]
            fp.close()

    # write binary and test A matrix and b vector
    if options.writeSystem:
        postfit.serialize_system(A,b,b_reference,w)

    print "Done."
