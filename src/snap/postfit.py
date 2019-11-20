#!/usr/bin/env python

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

import pickle, os
import clopts, training, lrun, ldeck, lrundeck
from clopts import options
from numpy import *
from scipy.linalg import lstsq
#from sklearn import linear_model, decomposition
import json

def serialize_system(A,b,b_reference,w):
    # Write out abtotal.dat (formatted for solution by llsw) and
    # Absystemd.dat, where A and b are serialized.
    print "Writing system (in human-readable form) to abtotal.dat."
    fp = open("abtotal.dat","w")
    for row in zip(A,b,w):
        for col in row[0]:
            fp.write("%g " % col)
        fp.write("%g %g\n" % (row[1],row[2]))
    fp.close()
    print "Serializing A and b to DumpSnap/Absystem.dat"
    file = "Absystem.dat"
    fp = open(options.dumpPath+os.sep+file,"w")
    p = pickle.Pickler(fp)
    p.dump((A,b,b_reference))
    fp.close()

def compute_mean_errors(A,b,SNAPCoeff,configList,configRows,groupRows,quantityRows):
    print "Computing mean errors."
    # helper to compute mean error over the indicated rows
    def compute_mean_error(A,b,coeff,rows):
        modelPerAtom = A[rows]*SNAPCoeff
        model = sum(modelPerAtom,axis=1)
        error = mean(abs(model - b[rows]))
        return error
    for quantity, func in zip( ("energy","force","virial"),
            ('eRows','fRows','vRows') ):
        errfp = open("SNAP%s_MeanError.dat"%quantity,"w")
#	conferr = open("SNAP%s_FullError.dat"%quantity,"w")
        # Loop over groups.
        for name, ind in sorted(groupRows.items()):
            rows = getattr(ind,func)
#	    print func, name, rows
            groupError = compute_mean_error(A,b,SNAPCoeff,rows)
            errfp.write("%s %12.8f\n" %(name,groupError))
#	    #Loop over configs in each group
#	    for ind, config in zip(configRows.items()):
    	    #for config, ind2 in sorted(configRows.items()):
#	        rows = getattr(ind,func)
#                configError = compute_mean_error(A,b,SNAPCoeff)
#	        conferr.write("%s %s %12.8f\n"%(name,config,configError))
        rows = getattr(quantityRows,func)
        totalError = compute_mean_error(A,b,SNAPCoeff,rows)
        errfp.write("\n%s error (mean of absolute values) %12.8f\n" % (quantity,totalError))
        errfp.close()
def compute_pca(A,b,w,SNAPCoeff,configList,configRows,groupRows,quantityRows):
	pca = decomposition.PCA(copy=True, iterated_power='auto', n_components=int(options.PCAsize), random_state=None,svd_solver='auto', tol=0.0, whiten=True)
	X = pca.fit(A*w[:,newaxis]).transform(A*w[:,newaxis])
	fp = open("Training_PCA.dat","w")
	for quantity, func in zip( ("energy","force","virial"), ('eRows','fRows','vRows') ):
	        for name, ind in sorted(groupRows.items()):
        	    rows = getattr(ind,func)
	            for row in zip(X[rows]):
			    fp.write("%s %s " % (func, name))
         		    for col in row[0]:
		                fp.write(" %g " % col)
			    fp.write("\n")
	fp = open("PCA.dat","w")
	for row in zip(pca.components_):
        	for col in row[0]:
	            fp.write("%g " % col)
		fp.write("\n")
	print "Top 5 PCA Fractions: "
	for c in range (0,min(5,int(options.PCAsize))):
		print c+1,pca.explained_variance_ratio_[c]

def write_detailed_results(A,b_training,b_reference,SNAPCoeff,configList,configRows,quantityRows):
    print "Writing training errors"
    # Compute model predictions and errors
    model = sum(A*SNAPCoeff,axis=1)
    error = b_training - b_reference - model
    # Stan's output
    for quantity, func in zip( ("energy","force","virial"),
            ('eRows','fRows','vRows') ):
        errfp = open("SNAP%s_ErrTrain.dat"%quantity,"w")
        rows = getattr(quantityRows,func)
        errfp.write("# %s, %s, %s, %s\n" % ("Train","Ref","Model=A.x","Training-Ref-Model"))
        errfp.write("# JSON Units, unweighted, energy scaled by nAtoms\n")
        for (valt,valr,valm,err) in zip(b_training[rows],b_reference[rows],model[rows],error[rows]):
            errfp.write("%12.12f %12.12f %12.12f %12.12f\n" % (valt,valr,valm,err))
        errfp.close()
    # json formatted predictions (model * numAtoms + b_reference)
    # Only energy rows are scaled by invatoms.
    for i, c in enumerate(configList):
        model[i] *= c.nAtoms
    model += b_reference
    configP = [] # Predictions data structure
    for i, c in enumerate(configList):
        configP.append( {"name": c.fullPath,
                            "index": c.lammpsIndex,
                            "energy": model[i],
                            "forces": model[configRows[i].fRows].tolist(),
                            "virials": model[configRows[i].vRows].tolist()})
    pfp = open("predictions.json","w")
    json.dump(configP,pfp)
    pfp.close()

