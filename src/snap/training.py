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

import json
import glob
import os
import pickle
from math import sqrt
from numpy import *
from snapexception import SNAPException
from clopts import options

class TrainingException(SNAPException):
    pass

MACHINE_TOL = 1.0e-10
outputgroupfilename = "grouplist.out"
#jsondir = "../../examples/Ta/JSON/"
#datadir = "Data"
#jsondir = options.jsonPath
#datadir = options.dataPath
floatformat = "%20.15g"
intformat = "%10d"
qdummy = 0.0
moldummy = 0
magdummy = 2.2
dirdummy = 1.0

###############################################################################
# A few light-weight classes for storing information about the training set
# data (configurations, energies, forces, virials). Consider replacing
# some of these with named tuples.

class GroupWeights(object):
    def __init__(self,eweight=1.0,fweight=1.0,vweight=1.0,gweight=1.0,
            numConfigs=0):
        self.eweight = eweight # weights for energies, forces, and virials
        self.fweight = fweight
        self.vweight = vweight
        self.gweight = gweight # group weights. Unused for now.
        self.numConfigs = numConfigs

class GroupRows(object):
    # One instance per group. Instantiate with row indices for the group
    # The getters return a numpy array which can be used for array-based
    # indexing in A and b.
    def __init__(self,eStartRow=0,nErows=0, fStartRow=0,nFrows=0, vStartRow=0):
        self.eRows = slice(eStartRow,eStartRow+nErows)
        self.fRows = slice(fStartRow,fStartRow+nFrows)
        self.vRows = slice(vStartRow,vStartRow + 6*nErows)

QuantityRows = GroupRows # We want the exact same functionality as provided by
                            # GroupRows, but a distinct name for the sake of
                            # clarity.

class ConfigRows(object):
    # One instance per configuration. Instantiate with row indices for the config.
    # The getters return a numpy array which can be used for array-based
    # indexing in A and b.
    def __init__(self,eRow=0,fStartRow=0,nFrows=0,vStartRow=0):
        self.eRow = eRow
        self.fRows = slice(fStartRow,fStartRow+nFrows)
        self.vRows = slice(vStartRow,vStartRow + 6)

class Config(object):
    def __init__(self,group="",energy=0.0,forces=None,virials=None,positions=None,
                types=None,cell=None,name=None):
        self.group = group
        self.fullPath = name
        self.name = os.path.basename(name)
        self.energy = energy
        self.forces = forces
        self.virials = zeros(6)
        self.virials[:3] = diag(virials)
        self.virials[3] = virials[1][2]
        self.virials[4] = virials[0][2]
        self.virials[5] = virials[0][1]
        self.positions = positions
        self.cell = cell
        self.types = types
        self.nAtoms = self.positions.shape[0] # number of rows
        self.volume = linalg.det(self.cell)

    def lammps_index(self,fNumber):
        self.lammpsIndex = fNumber # index of LAMMPS Data, log, and dump files
                                # e.g. Data/data.lammps_001

def read_group_weights(configs=None):
    print "Reading group weights from",options.groupFile
    try:
        fp = open(options.groupFile,"r") # TODO: add more informative exception
    except IOError:
        raise TrainingException("Error: Unable to open %s." % \
                options.groupFile)
    groups = {}
    for line in fp:
        if '#' in line or line.strip() is "": # exclude comments and empty lines
            continue
        try:
            name, size, ew, fw, vw = line.split()
            groups[name]=GroupWeights(numConfigs=int(size), eweight=float(ew),
                    fweight=float(fw), vweight=float(vw))
        except ValueError:
            raise TrainingException("Error: Syntax error in " + \
                    "%s" % options.groupFile + "; wrong number of values " + \
                    "or values of wrong type.")
    fp.close()
    # Verify the config counts and group names that were read in
    forCheck = dict.fromkeys(groups,0)
    for config in configs:
        try:
            forCheck[config.group] += 1
        except KeyError:
            raise TrainingException("Error: Group \"%s\" " % config.group + \
                    "present in training set, but has no entry in groups "
                    "file.")
    for group in groups.iterkeys():
        if groups[group].numConfigs != forCheck[group]:
#            print "Warning: Mismatch in number of " + \
#                    "configurations between the groups file and training " + \
#                    "set for group \"%s\"" % group)
            raise TrainingException("Error: Mismatch in number of " + \
                    "configurations between the groups file and training " + \
                    "set for group \"%s\"" % group)
    return groups

def construct_indices(configList):
    # Determine total number of energy, force, and virial rows in order to allocate
    # space for b_training. number of rows equals number of energies (1/config) +
    # number of virial components (6/config) + number of force components (3/atom).
    numErows = len(configList)
    numFrows = sum([3*c.nAtoms for c in configList])
    numVrows = 6*numErows
    b_training = zeros(numErows+numFrows+numVrows)
    # Construct List of ConfigRows, dictionary of GroupRows, b_training, and
    # quantityRows.
    # A key assumption made by the following code is that the configs are ordered
    # by group.
    fctr = numErows # index of first empty force row
    vctr = fctr + numFrows # index of first empty virial row
    quantityRows = QuantityRows(eStartRow=0, nErows=numErows, fStartRow=fctr,
            nFrows=numFrows,vStartRow=vctr)
    eGroupStart = 0 # index of first energy for current group
    fGroupStart = fctr # index of first force component for current group
    vGroupStart = vctr # index of first virial component for current group
    groupRowsList = {}
    configRowsList = []
    currentGroup = configList[0].group
    for i, config in enumerate(configList):
        nConfigFrows = 3*config.nAtoms
        configRowsList.append(ConfigRows(eRow=i, fStartRow=fctr, nFrows=nConfigFrows,
                                vStartRow=vctr))
        tc = configRowsList[-1] # for convenience. tc = "this config"
        b_training[tc.eRow] = config.energy
        b_training[tc.fRows] = config.forces.flatten()
        b_training[tc.vRows] = config.virials # /config.volume # TODO verify

        if config.group != currentGroup: # If true, loop has advanced to the next group
            groupRowsList[currentGroup] = GroupRows(eStartRow=eGroupStart,
                    nErows=i-eGroupStart,
                    fStartRow=fGroupStart,
                    nFrows=fctr-fGroupStart,
                    vStartRow=vGroupStart)
            currentGroup = config.group
            eGroupStart = i
            fGroupStart = fctr
            vGroupStart = vctr
        fctr += nConfigFrows
        vctr += 6
    # loop exits without adding final group. Add it here.
    groupRowsList[currentGroup] = GroupRows(eStartRow=eGroupStart,
            nErows=len(configList)-eGroupStart,
            fStartRow=fGroupStart,
            nFrows=fctr-fGroupStart,
            vStartRow=vGroupStart)

    return b_training, configRowsList, groupRowsList,quantityRows

def read_training_set(file="trainingset.dat"):
    print "Reading in training set data."
    # check for the existence of the file.
    fullPath = options.dataPath+os.sep+file
    foundPickle = True
    try:
        fp = open(fullPath,"r")
    except:
        foundPickle = False

    if foundPickle:
        print "Previously converted JSON data found."
        p = pickle.Unpickler(fp)
        try:
            configList, dataStyles, \
                    configRows, groupRows, quantityRows, b_training = p.load()
        except (ValueError, KeyError):
            raise TrainingException("Error: Attempt to import previously " + \
                    "converted training set data in %s failed. " % fullPath + \
                    "It may have been generated by an incompatible version " + \
                    "of this script.")
        finally:
            fp.close()
    else:
        print "Parsing JSON data."
        configList, dataStyles = parse_JSON()
        write_lammps_data(configList)
        b_training, \
                configRows, \
                groupRows, quantityRows = construct_indices(configList)
        fp = open(options.dataPath+os.sep+file,"w")
        p = pickle.Pickler(fp)
        p.dump((configList,dataStyles,configRows,groupRows, quantityRows,b_training))
    # Write out datakey.dat if requested.
    if options.writeDataKey:
        fp = open(options.dataPath + os.sep + "datakey.dat","w")
        nFormat = len("%d" % len(configList))
        dataFormat = "%%0%dd" % nFormat
	fp.write("%s\n" % ("# Group_Name"))
        for c in configList:
            indexString = dataFormat % c.lammpsIndex
            fp.write("%s /%s:%s\n" % (c.group,c.name,indexString))
        fp.close()
    return configList, dataStyles, configRows, groupRows, quantityRows, b_training

def convert_JSON(file="trainingset.dat"):
    configList, dataStyles = parse_JSON()
    write_lammps_data(configList)
    b_training, configRows, groupRows, quantityRows = construct_indices(configList)
    fullPath = options.dataPath+os.sep+file
    try:
        fp = open(fullPath,"w")
    except (OSError,IOError):
        raise TrainingException("Error: Attempt to open %s failed." % fullPath)
    p = pickle.Pickler(fp)
    p.dump((configList,configRows,groupRows,quantityRows,b_training))
    fp.close()

def getgroup(dir):
    name = dir.split("/")[-1]
    files  = glob.glob(dir+"/*.json")
    files.sort()
    size = len(files)
    group = {
        "name":name,
        "size":size,
        "files":files}
    return group

def printgroup(group):
    print "group name = ",group["name"]
    print "group size = ",group["size"]
    print "group files:"
#    for file in group["files"]:
#        print file
#    print

# Extract data from JSON files

#def parse_JSON(jsondir=options.jsonPath):
def parse_JSON():
    mkdir_p(options.dataPath)
    dirlist = glob.glob(options.jsonPath + "/*")
    if len(dirlist) == 0:
        raise TrainingException("Error: JSON empty or non-existent")
    dirlist.sort()

    grouplist = []
    for dir in dirlist:
        grouplist.append(getgroup(dir))

    outputgroupfile = open(outputgroupfilename,'w')
    print >>outputgroupfile,"# name size"
    for group in grouplist:
        print >>outputgroupfile,"%s %d" % (group["name"],group["size"])

    iframe = 0
    configList = []

    # default all JSON styles to LAMMPS metal units

    dataStyles = {}
    dataStyles['StressStyle'] = "bar"
    dataStyles['LatticeStyle'] = "angstrom"
    dataStyles['EnergyStyle'] = "electronvolt"
    dataStyles['ForceStyle'] = "electronvoltperangstrom"

    # unset flags signalling default overriden

    stressstyleflag = 0
    latticestyleflag = 0
    energystyleflag = 0
    forcestyleflag = 0

    for group in grouplist:
        print "\nGroup '%s' -- (%d files found)" % (group['name'],group['size'])
        for filename in group["files"]:
#            print filename
            try:
                jfile = open(filename,"r")
            except (OSError,IOError):
                raise TrainingException("Error: Unable to open %s" % filename)
            jfile.readline() # read the comment line
            try:
                jobj = json.load(jfile)
            except ValueError as e:
                raise TrainingException("Error: While attempting to read " + \
                        "%s, JSON parser reports: %s" %(filename, e.args[0]))
            try:
                config = jobj['Dataset']['Data'][0]
            except (TypeError, KeyError):
                raise TrainingException("Error: Extraction of ['Dataset']" + \
                        "['Data'][0] from %s failed." % filename)
            configKeys = ('NumAtoms','Lattice','AtomTypes','Positions','Forces',
                    'Energy','Stress')
            for k in configKeys:
                if not config.has_key(k):
                    raise TrainingException("Error: %s lacks " % filename + \
                            "required field \'%s\'." % k)
            # read the unit styles for key quantities, currently just stress
            # should be able to do this for all quantities with a loop over dataStyles.keys()[]
            strtmp = jobj['Dataset'].get('StressStyle',None)
            if strtmp != None:
                if stressstyleflag == 0:
                    dataStyles['StressStyle'] = strtmp
                    stressstyleflag = 1
                else:
                    if strtmp != dataStyles['StressStyle']:
                        raise TrainingException("Error: StressStyle %s " % strtmp + \
                                                "in file %s " % filename + \
                                                "does not match previous non-default StressStyle %s." % stressstyle)


            natoms = config['NumAtoms']
            types = config['AtomTypes']
            # This is the cell read in from JSON file
            # Copy values in to cellqm matrix
            # The values are listed as
            # [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
            # so we need to take transpose to
            # get [ax,ay,az] to be a column
            cellqm = array(config['Lattice']).transpose()
            # Inverse cell is [h k l]^t/V
            cellqminv = linalg.inv(cellqm)
            # This is the LAMMPS cell
            cell = lammps_cell(cellqm)
            cellprod = dot(cell,cellqminv)
            if natoms != len(config['Positions']):
                print "WARNING: natoms in %s does not match " % filename + \
                        "number of atoms in 'Positions' list."
                natoms = len(config['Positions'])
            xraw = array(config['Positions'])
            x = dot(cellprod,xraw.transpose()).transpose()
            f0 = array(config['Forces'])
            f = dot(cellprod,f0.transpose()).transpose()
            cell_flip(cell)
            energy = config['Energy']
            stress0 = array(config['Stress'])
            stress = dot(dot(cellprod,stress0),cellprod.transpose())
            configList.append(Config(group=group["name"], energy=energy,
                forces=f, virials=stress, positions=x,types=types, cell=cell,name=filename))

            iframe += 1
    # configList is completely populated. For load balancing in lammps
    # multiprocessing, index the configurations by sorting by number of atoms
    # and atomic density in descending order. These indices will be used in
    # log and dump file names. The indices are 1-based instead of 0- because
    # uloop style variables in the lammps input are are 1-based.
    def get_nAtoms(c): return c.nAtoms
    for i, c in enumerate(sorted(configList,key=get_nAtoms,reverse=True)):
        c.lammps_index(i+1)
    return configList, dataStyles

# Apply periodic shift to lattice vectors, if necessary
# We do this after rotation, as it should have no effect
# on the positions and forces.def cell_flip(cell):

def cell_flip(cell):

    # Check that yz is not too large for LAMMPS

    if abs(cell[1][2]) > 0.5*cell[1][1]:
        if cell[1][2] < 0.0:
            cell[1][2] += cell[1][1];
            cell[0][2] += cell[0][1];
        else:
            cell[1][2] -= cell[1][1];
            cell[0][2] -= cell[0][1];

    # Check that xz is not too large for LAMMPS

    if abs(cell[0][2]) > 0.5*cell[0][0]:
        if cell[0][2] < 0.0:
            cell[0][2] += cell[0][0];
        else:
            cell[0][2] -= cell[0][0];

    # Check that xy is not too large for LAMMPS

    if abs(cell[0][1]) > 0.5*cell[0][0]:
        if cell[0][1] < 0.0:
            cell[0][1] += cell[0][0];
        else:
            cell[0][1] -= cell[0][0];

    return cell

def lammps_cell(cellqm):

    cell = zeros((3,3))

    # Compute edge lengths

    cellqmtrans = cellqm.transpose()
    avec = cellqmtrans[0]
    bvec = cellqmtrans[1]
    cvec = cellqmtrans[2]
    anorm = sqrt((avec**2.0).sum())
    bnorm = sqrt((bvec**2.0).sum())
    cnorm = sqrt((cvec**2.0).sum())
    ahat = avec/anorm

    # Inverse cell is [h k l]^t/V

    cellqminv = linalg.inv(cellqm)

    lvec = cellqminv[2]
    lnorm = sqrt((lvec**2.0).sum())
    lhat = lvec*(1.0/lnorm)

    # ax = |A|

    cell[0][0] = anorm

    # bx = |A|.|B|/|A|
    # by = Sqrt(|B|^2 - bx^2)

    cell[0][1] = dot(ahat,bvec)
    cell[1][1] = sqrt(bnorm**2 - cell[0][1]**2)

    # cx = |A|.|C|/|A|
    # cy = (|B||C| - bx*cx)/by
    # cz = Sqrt(C^2 - cx^2 - cy^2 +cz^2)

    cell[0][2] = dot(ahat,cvec)
    cell[1][2] = (dot(bvec,cvec) - cell[0][1]*cell[0][2])/cell[1][1]
    cell[2][2] = sqrt(cnorm**2 - cell[0][2]**2 - cell[1][2]**2)

    return cell

# Generate data file from natoms,cell,x

def write_lammps_data(configList):

    nFiles = len(configList)
    # Construct filename format for data file
    nFormat = len("%d" % nFiles)
    dataFormat = "%%s/data.lammps_%%0%dd" % nFormat
    cellFormat = """   0.0 %s xlo xhi
        0.0 %s ylo yhi
        0.0 %s zlo zhi
        %s %s %s xy xz yz
        """ % (floatformat,floatformat,floatformat,
               floatformat,floatformat,floatformat)
    if (options.atomstyle == 'charge'):
        atomFormat  = "%s %s %s %s %s %s" % (intformat,intformat,floatformat,floatformat,floatformat,floatformat)
    elif (options.atomstyle == 'atomic'):
        atomFormat  = "%s %s %s %s %s" % (intformat,intformat,floatformat,floatformat,floatformat)
    elif (options.atomstyle == 'full'):
        atomFormat  = "%s %s %s %s %s %s %s" % (intformat,intformat,intformat,floatformat,floatformat,floatformat,floatformat)
    elif (options.atomstyle == 'spin'):
        atomFormat  = "%s %s %s %s %s %s %s %s %s" % (intformat,intformat,floatformat,floatformat,floatformat,floatformat,floatformat,floatformat,floatformat)
    if options.numTypes == 1:
        typedict = {options.type1:1,"1":1}
    elif options.numTypes == 2:
        typedict = {options.type1:1, options.type2:2, "1":1, "2":2}
    elif options.numTypes == 3:
        typedict = {options.type1:1, options.type2:2, options.type3:3, "1":1, "2":2, "3":3}
    else:
        typedict = {options.type1:1, options.type2:2, options.type3:3, options.type4:4, "1":1, "2":2, "3":3, "4":4}
    for ci, c in enumerate(configList):
        datafilename = dataFormat % (options.dataPath,c.lammpsIndex)
        datafile = open(datafilename,'w')
        print >>datafile,"""Comment line

              %d atoms
               0 bonds
               0 angles
               0 dihedrals
               0 impropers

              %d atom types
               0 bond types
               0 angle types
               0 dihedral types
               0 improper types
""" % (c.nAtoms,options.numTypes)
        print >>datafile, cellFormat % (c.cell[0][0],c.cell[1][1],
                c.cell[2][2],c.cell[0][1],c.cell[0][2],c.cell[1][2])
        print >>datafile,"Atoms\n"
        for i in xrange(c.nAtoms):
            try:
                if (options.atomstyle == 'charge'):
                    print >>datafile,atomFormat % (i+1,typedict[c.types[i]],qdummy,
                        c.positions[i][0],c.positions[i][1],c.positions[i][2])
                elif (options.atomstyle == 'atomic'):
                    print >>datafile,atomFormat % (i+1,typedict[c.types[i]],
                        c.positions[i][0],c.positions[i][1],c.positions[i][2])
                elif (options.atomstyle == 'full'):
                    print >>datafile,atomFormat % (i+1,moldummy,typedict[c.types[i]],qdummy,
                        c.positions[i][0],c.positions[i][1],c.positions[i][2])
                elif (options.atomstyle == 'spin'):
                    print >>datafile,atomFormat % (i+1,typedict[c.types[i]],magdummy,
                        c.positions[i][0],c.positions[i][1],c.positions[i][2],0,0,dirdummy)
            except KeyError:
                raise TrainingException("Error: Encountered unknown type " + \
                        "%s while converting configuration " % c.types[i] + \
                        "%d." % ci)
        datafile.close()

def mkdir_p(path):
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
