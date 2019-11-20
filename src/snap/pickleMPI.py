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

import pickle, os, numpy, sys
import clopts, training, lrun, ldeck, lrundeck, postfit, ltest
from clopts import options


fullPath = sys.argv[1]
fp = open(fullPath,"r")
print "Previously generated A and b (Absystem.dat) found."
p = pickle.Unpickler(fp)
try:
	A, b, b_reference = p.load()
	numpy.save('A_pickle.out',A)
	numpy.save('b_pickle.out',b)
	numpy.save('b_reference_pickle.out',b_reference)

except (ValueError, KeyError):
	print "Warning: Attempt to import previously " + \
	"generated A and b from %s failed. Will now " % fullPath + \
	"attempt to parse existing LAMMPS results."
	foundAbPickle = False
finally:
	fp.close()

