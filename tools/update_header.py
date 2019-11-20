#!/usr/bin/python 

# user settings

legacyflag = True
headerfilename = "HEADER"
#targetfilelist = ["foo*.py"]
targetfilelist = ["src/*.py","src/snap/*.py"]

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

# Additional authors: 
# Mary Alice Cusentino
# Steve Plimpton
# Peter Schultz
# Adam Stephens
# Laura Swiler
# Mitchell Wood

# ----------------------------END-HEADER-------------------------------------

import glob

beginsentinelline = \
"# ---------------------------BEGIN-HEADER------------------------------------\n"
endsentinelline = \
"# ----------------------------END-HEADER-------------------------------------\n"

# remove old header, add new header

def update_header(targetfilename):
    # read file

    targetfile = open(targetfilename,'r')
    targetlines = targetfile.readlines()
    targetfile.close()

    # rewrite file with header

    targetfile = open(targetfilename,'w')
    toggle = True
    for line in targetlines:
        if line == beginsentinelline:
            toggle = not toggle
            print >>targetfile, line,
            for line in headerlines:
                print >>targetfile, line,
        elif line == endsentinelline:
            toggle = not toggle
        if toggle:
            print >>targetfile, line,
    targetfile.close()

# optional insert sentinel lines into legacy files

def insert_sentinel_lines(targetfilename):

    # first instance of these lines will be used
    # as insertion points for sentinel lines

    insertbegin = "# Copyright (2016) Sandia Corporation. \n"
    insertend = "# Laura Swiler\n"

    # read file

    targetfile = open(targetfilename,'r')
    targetlines = targetfile.readlines()
    targetfile.close()

    # rewrite file with sentinel lines
 
    targetfile = open(targetfilename,'w')

    needbegin = True
    needend = True
    for line in targetlines:
        if line == insertbegin and needbegin:
            print >>targetfile, beginsentinelline,
            print >>targetfile, line,
            needbegin = False
        elif line == insertend and needend:
            print >>targetfile, line,
            print >>targetfile, endsentinelline,
            needend = False
        else:
            print >>targetfile, line,

    targetfile.close()

# main program

targetfiles = []
for file in targetfilelist:
    targetfiles += glob.glob(file)

headerfile = open(headerfilename,'r')
headerlines = headerfile.readlines()
headerfile.close()

for targetfilename in targetfiles:
    print targetfilename

    # optional insert sentinels into legacy files
    
    if (legacyflag):
        insert_sentinel_lines(targetfilename)

    update_header(targetfilename)

