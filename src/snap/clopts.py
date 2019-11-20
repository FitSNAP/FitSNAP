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

import sys
from optparse import OptionParser
from inparse import InFileParser
from snapexception import SNAPException

class SNAPOptsError(SNAPException):
    pass

# Command Line Options:
# --convert-JSON <type1> <type2>
# --input,-i file_name
# Otherwise, input keywords will be looked for on stdin: ./fitsnap < file_name

# A class to contain snap options for access everywhere else.
# The update in set_options is a trick. By modifying the dictionary directly,
# we expose the contents of optionsDict as class members.
class SNAPOptions(object):
    def __init__(self):
        pass

    def set_options(self,optionsDict):
        self.__dict__.update(optionsDict)

# Module-level variable which contains all program options
options = SNAPOptions()

# Define the command line parser
_usage = "usage: %prog [options] [argument list]"
_clParser = OptionParser(_usage)
_clParser.add_option("--convert-JSON", "-c", action="store_true",
        dest="convertJSON", default=False,help="Convert JSON and exit. " + \
        "User must specify the element labels as additional arguments.")
_clParser.add_option("--input","-i", action="store",dest="inputFile",
        help="Input file name.")
_clParser.add_option("--verbosity","-v", action="store",type='int',
            dest="verbosity",default="1",
            help="Verbosity of output. Currently does nothing.",
            metavar="N")

### Define the input file parser
# First, we need a set of functions to test ranges. These functions accept the
# user provided keyword value as an argument. They return a string if the
# variable is outside of acceptable range, and None otherwise.
def _range_numTypes(v):
    if v <= 4:
        return None
    return "numTypes is required to be less than 4."

def _range_numConsts(v):
    if v <= 4:
        return None
    return "numConsts is required to be less than numTypes"

def _range_positive(v):
    if v > 0:
        return None
    return "something is <= 0 that should be positive!"


def _range_positive_even(v):
    if v > 0 and v % 2 == 0:
        return None
    return "something is <=0 or odd that should be positive and even!"


def _range_nonnegative(v):
    if v >= 0:
        return None
    return "something is < 0 that should be >= 0!"

def _range_solver(v):
    if v == "SVD" or v == "LASSO" or v == "RIDGE" or v == "ELASTIC":
        return None
    return "solver is required to be SVD, LASSO, RIDGE or ELASTIC."

def _range_atypes(v):
    if v.lower() == "atomic" or v.lower() == "charge" or v.lower() == "spin" or v.lower() == "full":
        return None
    return "Atom Type is required to be atomic, charge, spin or full."

def _range_onoff(v):
    if v == 0 or v == 1:
        return None
    return "value must be 0 or 1."

# Define requirements. These functions behave the same as the range functions,
# but accept a dictionary of {"keyword":value} pairs.

# For a pure material (numTypes=1), require these keywords :
_pureKeys  = set(("twojmax","rcutfac","rfac0","rmin0",
        "zblcutinner","zblcutouter","type1","zblz1","radelem1","atomstyle"))

# For a binary material (numTypes=2), require these keywords:
_binaryKeys = set(("twojmax","rcutfac","rfac0","rmin0",
        "zblcutinner","zblcutouter","qcoul","rcoul","freezeold",
        "type1","type2","zblz1","zblz2","wj1","wj2","radelem1","radelem2","atomstyle"))
_ternaryKeys = set(("twojmax","rcutfac","rfac0","rmin0",
        "zblcutinner","zblcutouter","qcoul","rcoul","freezeold",
        "type1","type2","type3","zblz1","zblz2","zblz3","wj1","wj2","wj3",
        "radelem1","radelem2","radelem3","atomstyle"))
_quadKeys = set(("twojmax","rcutfac","rfac0","rmin0",
        "zblcutinner","zblcutouter","qcoul","rcoul","freezeold",
        "type1","type2","type3","type4","zblz1","zblz2","zblz4","zblz3",
        "wj1","wj2","wj3","wj4","radelem1","radelem2","radelem3","radelem4","atomstyle"))

def _required_numTypes(defined):
    if defined["numTypes"] == 1:
        missing = _pureKeys.difference(defined)
        if missing:
            return "Missing keywords for pure material: %s" \
                    % " ".join(list(missing)) + "."
        else: return None
    elif defined["numTypes"] == 2:
        missing = _binaryKeys.difference(defined)
        if missing:
            return "Missing keywords for binary material: %s" \
                    % " ".join(list(missing)) + "."
        else: return None
    elif defined["numTypes"] == 3:
        missing = _ternaryKeys.difference(defined)
        if missing:
            return "Missing keywords for ternary material: %s" \
                    % " ".join(list(missing)) + "."
        else: return None
    else:
        missing = _quadKeys.difference(defined)
        if missing:
            return "Missing keywords for quadnary material: %s" \
                    % " ".join(list(missing)) + "."
        else: return None


_fileParser = InFileParser()
_fileParser.add_keyword(keyword="numTypes",default=1,range=_range_numTypes,required=_required_numTypes)
_fileParser.add_keyword(keyword="twojmax",default=6,range=_range_positive_even)
_fileParser.add_keyword(keyword="rfac0", default=0.99363,range=_range_positive)
_fileParser.add_keyword(keyword="rmin0", vtype=float, default=0.0)
_fileParser.add_keyword(keyword="zblcutinner", vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="zblcutouter", vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="atomstyle",vtype=str,default="charge",range=_range_atypes)
_fileParser.add_keyword(keyword="qcoul",vtype=float,range=_range_positive,default=0.0)
_fileParser.add_keyword(keyword="rcoul",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="zblz1",vtype=int,range=_range_nonnegative)
_fileParser.add_keyword(keyword="zblz2",vtype=int,range=_range_nonnegative)
_fileParser.add_keyword(keyword="zblz3",vtype=int,range=_range_nonnegative)
_fileParser.add_keyword(keyword="zblz4",vtype=int,range=_range_nonnegative)
_fileParser.add_keyword(keyword="PairInclude",vtype=str,default="default_pair.inc")
_fileParser.add_keyword(keyword="rcutfac",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="type1",vtype=str)
_fileParser.add_keyword(keyword="type2",vtype=str)
_fileParser.add_keyword(keyword="type3",vtype=str)
_fileParser.add_keyword(keyword="type4",vtype=str)
_fileParser.add_keyword(keyword="wj1",vtype=float)
_fileParser.add_keyword(keyword="wj2",vtype=float)
_fileParser.add_keyword(keyword="wj3",vtype=float)
_fileParser.add_keyword(keyword="wj4",vtype=float)
_fileParser.add_keyword(keyword="radelem1",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="radelem2",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="radelem3",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="radelem4",vtype=float,range=_range_positive)
_fileParser.add_keyword(keyword="Eshift1",vtype=float, default=0.00000000000)
_fileParser.add_keyword(keyword="Eshift2",vtype=float, default=0.00000000000)
_fileParser.add_keyword(keyword="Eshift3",vtype=float, default=0.00000000000)
_fileParser.add_keyword(keyword="Eshift4",vtype=float, default=0.00000000000)
_fileParser.add_keyword(keyword="gamma",default=1.0)
_fileParser.add_keyword(keyword="solver",default="SVD",range=_range_solver)
_fileParser.add_keyword(keyword="lammpsPath",default="./lmp.exe")
_fileParser.add_keyword(keyword="groupFile",default="grouplist.in")
_fileParser.add_keyword(keyword="maxConcurrency",default=1,range=_range_nonnegative)
_fileParser.add_keyword(keyword="mpiLauncherLAMMPS",default="mpiexec")
_fileParser.add_keyword(keyword="mpiLauncherLSTSQ",default="mpiexec -np 1")
_fileParser.add_keyword(keyword="verifyConfigs",default=1,range=_range_onoff)
_fileParser.add_keyword(keyword="writeSystem",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="writeTrainingErrors",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="computeTestingErrors",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="staleOutputCheck",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="numConstants",default=0,range=_range_numConsts)
_fileParser.add_keyword(keyword="writeDataKey",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="runLammps",default=1,range=_range_onoff)
_fileParser.add_keyword(keyword="potentialFileName",default="")
_fileParser.add_keyword(keyword="normweight",default="-12")
_fileParser.add_keyword(keyword="normratio",default="0.50")
_fileParser.add_keyword(keyword="bzeroflag",default=1,range=_range_onoff)
_fileParser.add_keyword(keyword="quadratic",default=0,range=_range_onoff)
_fileParser.add_keyword(keyword="PCAsize",vtype=int,range=_range_nonnegative,default=0)
_fileParser.add_keyword(keyword="jsonPath",vtype=str,default="JSON")
_fileParser.add_keyword(keyword="dataPath",vtype=str,default="Data")
_fileParser.add_keyword(keyword="dumpPath",vtype=str,default="DumpSnap")
_fileParser.add_keyword(keyword="dumpPathTest",vtype=str,default="DumpTest")

_fileParser.add_keyword(keyword="freezeold",vtype=int,range=_range_nonnegative,default=0)
_fileParser.add_keyword(keyword="coeffold",default="")
_fileParser.add_keyword(keyword="paramold",default="")

## Last 3 are additions by MAW 3/7/17 to enable SNAP as a reference potential.
## The advantage will be to 'freeze' bispectrum weights for multi-elem systems

###############################################################################
## This is the one user-callable function for the module. The argument is a
## list of command line arguments (i.e. sys.argv)
def parse_snap_options(optionList):
    clOptions, clArgs = _clParser.parse_args(optionList[1:])
    optionsDict = {"convertJSON":False}
    if clOptions.convertJSON:
        optionsDict["convertJSON"] = True
        if clOptions.inputFile:
            print "Warning: Input file ignored with --convert-JSON."
        numArgs = len(clArgs)
        if numArgs < 1 or numArgs > 2:
            _clParser.error("ERROR: --convert-JSON switch requires one or " + \
                    "two type names.")
        else:
            optionsDict["numTypes"] = numArgs
            if numArgs == 1:
                optionsDict["type1"] = clArgs[0]
            elif numArgs == 2:
                optionsDict["type1"] = clArgs[0]
                optionsDict["type2"] = clArgs[1]
            elif numArgs == 3:
                optionsDict["type1"] = clArgs[0]
                optionsDict["type2"] = clArgs[1]
                optionsDict["type3"] = clArgs[2]
            else:
                optionsDict["type1"] = clArgs[0]
                optionsDict["type2"] = clArgs[1]
                optionsDict["type3"] = clArgs[2]
                optionsDict["type4"] = clArgs[3]

    elif clOptions.inputFile:
        optionsDict.update(_fileParser.parse(clOptions.inputFile))
    else:
        print "Waiting for input on stdin...",
        sys.stdout.flush()
        optionsDict.update(_fileParser.parse())
        print "Got it!"
    options.set_options(optionsDict)
#    print "\nUser input --------------"
#    for k, v in sorted(optionsDict.items()):
#        print "%s = %s" % (k,str(v))
#    print "-------------------------"
