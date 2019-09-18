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

from snap import snap, clopts
from sys import argv, exit
from numpy import *
from snap.snapexception import SNAPException

print "Version %s\n" % snap.version

try:
    clopts.parse_snap_options(argv)
    if clopts.options.convertJSON:
        snap.training.convert_JSON()
    else:
        snap.calculate_SNAP()
except SNAPException as e:
    print e.message
    exit(1)

