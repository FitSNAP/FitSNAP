# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3 
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
# 
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
# 
# #### Original author: 
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps 
# 
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
# 
# #### Additional authors (alphabetical): 
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
#     
# <!-----------------END-HEADER------------------------------------->

insertsentinellines = False # only use for newly added files
headerfilename = "./tools/HEADER"

targetfilelist = ["./fitsnap3/*.py","./README.md", "./tools/*.py"]
#targetfilelist = ["./foo.py"]

import glob

beginsentinelline = \
"<!----------------BEGIN-HEADER------------------------------------>\n"
endsentinelline = \
"<!-----------------END-HEADER------------------------------------->\n"

# remove old header, add new header

def update_header(targetfilename):

    if targetfilename.find(".py") > -1:
        pythonflag = True
        pythoncomment = "# "
        beginsentinellinetmp = pythoncomment + beginsentinelline
        endsentinellinetmp = pythoncomment + endsentinelline
    else:
        pythonflag = False
        beginsentinellinetmp = beginsentinelline
        endsentinellinetmp = endsentinelline

    # read file

    targetfile = open(targetfilename,'r')
    targetlines = targetfile.readlines()
    targetfile.close()

    # rewrite file with header

    targetfile = open(targetfilename,'w')
    toggle = True
    for line in targetlines:
        if line == beginsentinellinetmp:
            toggle = not toggle
            targetfile.write(line)
            for line in headerlines:
                if pythonflag:
                    line = pythoncomment + line
                targetfile.write(line)
        elif line == endsentinellinetmp:
            toggle = not toggle
        if toggle:
            targetfile.write(line)
    targetfile.close()

# optional insert sentinel lines into new files

def insert_sentinel_lines(targetfilename):

    # first instance of these lines will be used
    # as insertion points for sentinel lines

    insertbegin = "# Copyright (2016) Sandia Corporation. \n"
    insertend = "# Mitchell Wood\n"

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
#        if needbegin:
            targetfile.write(beginsentinelline)
            targetfile.write(line)
            needbegin = False
        elif line == insertend and needend:
#        elif needend:
            targetfile.write(line)
            targetfile.write(endsentinelline)
            needend = False
        else:
            targetfile.write(line)

    targetfile.close()

# main program

targetfiles = []
for file in targetfilelist:
    targetfiles += glob.glob(file)

headerfile = open(headerfilename,'r')
headerlines = headerfile.readlines()
headerfile.close()

for targetfilename in targetfiles:
    print(targetfilename)

    # optional insert sentinels into new files

    if (insertsentinellines):
        insert_sentinel_lines(targetfilename)

    update_header(targetfilename)
