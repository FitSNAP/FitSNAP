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

# This module provides a wrapper to prevent log and dump from writing to stdout.

import sys
from contextlib import contextmanager
from StringIO import StringIO
import numpy as np # namespace clash between numpy and log
import log
import dump


# THis is pretty kludgey. I found it on stackoverflow. As stated above, the
# purpose is to prevent log and dump from writing to stdout. The decorator
# @contextmanager makes pizza_do work with the python "with" statement. The
# effect is that anything within a "with pizza_do():" block will have its
# standard output redirected.
@contextmanager
def _pizza_do():
    saved_stdout = sys.stdout # back up sys.stdout
    devnull = StringIO() # create a place for stdout to go
    sys.stdout = devnull # Redirect
    try:  # enclose in try/finally to ensure that sys.stdout is restored
        yield
    finally:
        sys.stdout = saved_stdout

def get_column_names(fileName):
    with _pizza_do():
        l = log.log(fileName)
        columnNames = l.names
    return columnNames

def process_log(fileName, betaLabels):

    with _pizza_do():
        l = log.log(fileName)
    energy, volume = [val[0] for val in l.get("c_sume","Volume")]
    virials = [val[0] for val in l.get("Pxx","Pyy","Pzz","Pyz","Pxz","Pxy")]
    virials = np.array(virials)
    betas = [val[0] for val in l.get(*betaLabels)]
    return energy, volume, virials, betas

def process_log_test(fileName):

    with _pizza_do():
        l = log.log(fileName)
    energy = l.get("PotEng")[0]
    virials = [val[0] for val in l.get("Pxx","Pyy","Pzz","Pyz","Pxz","Pxy")]
    virials = np.array(virials)
    return energy, virials

def process_dump(fileName):
    with _pizza_do():
        d = dump.dump(fileName)
    snap = d.snaps[0]
    atoms = np.array(snap.atoms)
    positions = atoms[:,2:5]
    types = np.array(atoms[:,1],dtype='int').tolist()
    forces = np.array(atoms[:,5:8])
    cell = np.zeros((3,3))
    cell[0][0] = snap.xhi - snap.xlo
    cell[1][1] = snap.yhi - snap.ylo
    cell[2][2] = snap.zhi - snap.zlo
    cell[0][1] = snap.xy
    cell[0][2] = snap.xz
    cell[1][2] = snap.yz
    return cell, positions, types, forces

def process_dump_test(fileName):
    with _pizza_do():
        d = dump.dump(fileName)
    snap = d.snaps[0]
    atoms = np.array(snap.atoms)
    types = np.array(atoms[:,1],dtype='int').tolist()
    forces = np.array(atoms[:,5:8])
    return types, forces

def process_dump_db(fileName):
    with _pizza_do():
        d = dump.dump(fileName)
    return np.array(d.snaps[0].atoms)
