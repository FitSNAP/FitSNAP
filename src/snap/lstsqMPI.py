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

import pickle, os, numpy
#import clopts, training, lrun, ldeck, lrundeck, postfit, ltest
#from clopts import options
from numpy import *
from scipy.linalg import lstsq
#from sklearn import linear_model, decomposition
#from snapexception import SNAPException
#import json

mypid = os.getpid()
print('Current host: ')
os.system('hostname')
A = numpy.load('A.out.npy')
b = numpy.load('b.out.npy')
w = numpy.load('w.out.npy')
#print(A.shape,b.shape)
print('Performing lstsq')
SNAPCoeff, res, rank, s = lstsq(A*w[:,newaxis],b*w)
print('Finished lstsq')
#output = open('SNAPCoeff.out','w')
#SNAPCoeff.tofile(output, sep=" ", format="%s")
#output.close()


os.system('touch SNAPCoeff.out.npy')
os.system('touch res.out.npy')
os.system('touch s.out.npy')


exists1 = os.path.isfile('SNAPCoeff.out.npy')
exists2 = os.path.isfile('res.out.npy')
exists3 = os.path.isfile('s.out.npy')

print('Whether lstsq files exist')
print(exists1, exists2, exists3)


numpy.save('SNAPCoeff.out',SNAPCoeff)
numpy.save('res.out',res)
numpy.save('s.out',s)


output = open('rank.out','w')
output.write('%s\n' % rank)
output.close()
