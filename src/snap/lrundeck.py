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

import os, sys
from clopts import options
from snapexception import SNAPException


class LRUNDECKException(SNAPException):
    pass
def _four_species(SNAPCoeff,numCoeffs,coeffindices):
    # Generate the potential file name if unspecified
    basename = options.potentialFileName
    if not basename:
        basename = options.type1 + options.type2 + options.type3 + options.type4
    modname = "pot_%s.mod" % basename
    paramname = "%s.snapparam" % basename
    coeffname = "%s.snapcoeff" % basename
    # Write mod file
    try:
        lmpoutput = open(modname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " % modname + \
                "writing.")

    print >>lmpoutput, "# Definition of SNAP+ZBL potential."
    print >>lmpoutput,"set type %1d charge %g" % (1,options.qcoul)
    print >>lmpoutput,"set type %1d charge %g" % (2,-options.qcoul)
    print >>lmpoutput,"set type %1d charge %g" % (3,options.qcoul)
    print >>lmpoutput,"set type %1d charge %g" % (4,-options.qcoul)
    print >>lmpoutput, "variable zblcutinner equal %g" % options.zblcutinner
    print >>lmpoutput, "variable zblcutouter equal %g" % options.zblcutouter
    print >>lmpoutput, "variable zblz1 equal %g" % options.zblz1
    print >>lmpoutput, "variable zblz2 equal %g" % options.zblz2
    print >>lmpoutput, "variable zblz3 equal %g" % options.zblz3
    print >>lmpoutput, "variable zblz4 equal %g" % options.zblz4
    print >>lmpoutput, "variable rcoul equal %g" % options.rcoul

    print >>lmpoutput, \
        """
# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay coul/long ${rcoul} &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff * * coul/long
pair_coeff 1 1 zbl ${zblz1} ${zblz1}
pair_coeff 1 2 zbl ${zblz1} ${zblz2}
pair_coeff 1 3 zbl ${zblz1} ${zblz3}
pair_coeff 1 4 zbl ${zblz1} ${zblz4}
pair_coeff 2 2 zbl ${zblz2} ${zblz2}
pair_coeff 2 3 zbl ${zblz2} ${zblz3}
pair_coeff 2 4 zbl ${zblz2} ${zblz4}
pair_coeff 3 3 zbl ${zblz3} ${zblz3}
pair_coeff 3 4 zbl ${zblz3} ${zblz4}
pair_coeff 4 4 zbl ${zblz4} ${zblz4}
pair_coeff * * snap %s.snapcoeff %s.snapparam %s %s %s %s
kspace_style ewald 1.0e-5
""" % (basename, basename, options.type1, options.type2, options.type3, options.type4)

    lmpoutput.close()
    # Write SNAP params file
    try:
        lmpoutput = open(paramname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " %paramname + \
                "writing.")
    print >>lmpoutput, "# required"
    print >>lmpoutput, "rcutfac %g" % options.rcutfac
    print >>lmpoutput, "twojmax %g" % options.twojmax
    print >>lmpoutput, "\n# optional\n"
#    print >>lmpoutput, "gamma %g" % options.gamma
    print >>lmpoutput, "rfac0 %g" % options.rfac0
    print >>lmpoutput, "rmin0 %g" % options.rmin0
    print >>lmpoutput, "bzeroflag %d"  % options.bzeroflag
    print >>lmpoutput, "quadraticflag %d \n"  % options.quadratic
    lmpoutput.close()

    # Write SNAP coefficients file
    fp = open(coeffname,"w")
    fp.write("# LAMMPS SNAP coefficients for %s%s%s\n\n" % (options.type1,
        options.type2, options.type3))
    fp.write("4 %d\n" % (numCoeffs+1,))
    fp.write("%s %g %g\n" %(options.type1,options.radelem1, options.wj1))
    for coeff in SNAPCoeff[:numCoeffs+1]:
            fp.write("%16.12f\n" % coeff)
    fp.write("%s %g %g\n" %(options.type2,options.radelem2, options.wj2))
#    if options.numConstants < options.numTypes:
#        fp.write("  0.000000000000\n")
    for coeff in SNAPCoeff[numCoeffs+1:2*numCoeffs+2]:
        fp.write("%16.12f\n" % coeff)
    fp.write("%s %g %g\n" %(options.type3,options.radelem3, options.wj3))
    for coeff in SNAPCoeff[2*numCoeffs+2:3*numCoeffs+3]:
            fp.write("%16.12f\n" % coeff)
    fp.write("%s %g %g\n" %(options.type4,options.radelem4, options.wj4))
    for coeff in SNAPCoeff[3*numCoeffs+3:]:
            fp.write("%16.12f\n" % coeff)
    fp.close()

def _three_species(SNAPCoeff,numCoeffs,coeffindices):
    # Generate the potential file name if unspecified
    basename = options.potentialFileName
    if not basename:
        basename = options.type1 + options.type2 + options.type3
    modname = "pot_%s.mod" % basename
    paramname = "%s.snapparam" % basename
    coeffname = "%s.snapcoeff" % basename
    # Write mod file
    try:
        lmpoutput = open(modname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " % modname + \
                "writing.")

    print >>lmpoutput, "# Definition of SNAP+ZBL potential."
    itype = 1
    print >>lmpoutput,"set type %1d charge %g" % (itype,options.qcoul)
    itype = 2
    print >>lmpoutput,"set type %1d charge %g" % (itype,-options.qcoul)
    itype = 3
    print >>lmpoutput,"set type %1d charge %g" % (itype,options.qcoul)
    print >>lmpoutput, "variable zblcutinner equal %g" % options.zblcutinner
    print >>lmpoutput, "variable zblcutouter equal %g" % options.zblcutouter
    print >>lmpoutput, "variable zblz1 equal %g" % options.zblz1
    print >>lmpoutput, "variable zblz2 equal %g" % options.zblz2
    print >>lmpoutput, "variable zblz3 equal %g" % options.zblz3
    print >>lmpoutput, "variable rcoul equal %g" % options.rcoul

    print >>lmpoutput, \
        """
# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay coul/long ${rcoul} &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff * * coul/long
pair_coeff 1 1 zbl ${zblz1} ${zblz1}
pair_coeff 1 2 zbl ${zblz1} ${zblz2}
pair_coeff 1 3 zbl ${zblz1} ${zblz3}
pair_coeff 2 2 zbl ${zblz2} ${zblz2}
pair_coeff 2 3 zbl ${zblz2} ${zblz3}
pair_coeff 3 3 zbl ${zblz3} ${zblz3}
pair_coeff * * snap %s.snapcoeff %s.snapparam %s %s %s
kspace_style ewald 1.0e-5
""" % (basename, basename, options.type1, options.type2, options.type3)

    lmpoutput.close()
    # Write SNAP params file
    try:
        lmpoutput = open(paramname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " %paramname + \
                "writing.")
    print >>lmpoutput, "# required"
    print >>lmpoutput, "rcutfac %g" % options.rcutfac
    print >>lmpoutput, "twojmax %g" % options.twojmax
    print >>lmpoutput, "\n# optional\n"
#    print >>lmpoutput, "gamma %g" % options.gamma
    print >>lmpoutput, "rfac0 %g" % options.rfac0
    print >>lmpoutput, "rmin0 %g" % options.rmin0
    print >>lmpoutput, "bzeroflag %d"  % options.bzeroflag
    print >>lmpoutput, "quadraticflag %d \n"  % options.quadratic
    lmpoutput.close()

    # Write SNAP coefficients file
    fp = open(coeffname,"w")
    fp.write("# LAMMPS SNAP coefficients for %s%s%s\n\n" % (options.type1,
        options.type2, options.type3))
    fp.write("3 %d\n" % (numCoeffs+1,))
    fp.write("%s %g %g\n" %(options.type1,options.radelem1, options.wj1))
    for coeff in SNAPCoeff[:numCoeffs+1]:
            fp.write("%16.12f\n" % coeff)
    fp.write("%s %g %g\n" %(options.type2,options.radelem2, options.wj2))
#    if options.numConstants < options.numTypes:
#        fp.write("  0.000000000000\n")
    for coeff in SNAPCoeff[numCoeffs+1:2*numCoeffs+2]:
        fp.write("%16.12f\n" % coeff)
    fp.write("%s %g %g\n" %(options.type3,options.radelem3, options.wj3))
    for coeff in SNAPCoeff[2*numCoeffs+2:]:
            fp.write("%16.12f\n" % coeff)
    fp.close()

def _two_species(SNAPCoeff,numCoeffs,coeffindices):
    # Generate the potential file name if unspecified
    basename = options.potentialFileName
    if not basename:
        basename = options.type1 + options.type2
    modname = "pot_%s.mod" % basename
    paramname = "%s.snapparam" % basename
    coeffname = "%s.snapcoeff" % basename
    # Write mod file
    try:
        lmpoutput = open(modname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " % modname + \
                "writing.")

    print >>lmpoutput, "# Definition of SNAP+ZBL potential."
    itype = 1
    print >>lmpoutput,"set type %1d charge %g" % (itype,options.qcoul)
    itype = 2
    print >>lmpoutput,"set type %1d charge %g" % (itype,-options.qcoul)
    print >>lmpoutput, "variable zblcutinner equal %g" % options.zblcutinner
    print >>lmpoutput, "variable zblcutouter equal %g" % options.zblcutouter
    print >>lmpoutput, "variable zblz1 equal %g" % options.zblz1
    print >>lmpoutput, "variable zblz2 equal %g" % options.zblz2
    print >>lmpoutput, "variable rcoul equal %g" % options.rcoul

    print >>lmpoutput, \
        """
# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay coul/long ${rcoul} &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff * * coul/long
pair_coeff 1 1 zbl ${zblz1} ${zblz1}
pair_coeff 1 2 zbl ${zblz1} ${zblz2}
pair_coeff 2 2 zbl ${zblz2} ${zblz2}
pair_coeff * * snap %s.snapcoeff %s.snapparam %s %s
kspace_style ewald 1.0e-5
""" % (basename, basename, options.type1, options.type2)

    lmpoutput.close()
    # Write SNAP params file
    try:
        lmpoutput = open(paramname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " %paramname + \
                "writing.")
    print >>lmpoutput, "# required"
    print >>lmpoutput, "rcutfac %g" % options.rcutfac
    print >>lmpoutput, "twojmax %g" % options.twojmax
    print >>lmpoutput, "\n# optional\n"
#    print >>lmpoutput, "gamma %g" % options.gamma
    print >>lmpoutput, "rfac0 %g" % options.rfac0
    print >>lmpoutput, "rmin0 %g" % options.rmin0
    print >>lmpoutput, "bzeroflag %d"  % options.bzeroflag
    print >>lmpoutput, "quadraticflag %d \n"  % options.quadratic
    lmpoutput.close()

    # Write SNAP coefficients file
    fp = open(coeffname,"w")
    fp.write("# LAMMPS SNAP coefficients for %s%s\n\n" % (options.type1,
        options.type2))
    fp.write("2 %d\n" % (numCoeffs+1,))
    fp.write("%s %g %g\n" %(options.type1,options.radelem1, options.wj1))
    line=0
    for coeff in SNAPCoeff[:numCoeffs+1]:
            fp.write("%16.12f # B%s \n"%(coeff, coeffindices[line]))
            line+=1
    fp.write("%s %g %g\n" %(options.type2,options.radelem2, options.wj2))
#    if options.numConstants < options.numTypes:
#        fp.write("  0.000000000000\n")
    line=0
    for coeff in SNAPCoeff[numCoeffs+1:]:
        fp.write("%16.12f # B%s \n"%(coeff, coeffindices[line]))
        line+=1
    fp.close()

def _one_species(SNAPCoeff,numCoeffs,coeffindices):
    # Generate the potential file name if unspecified
    basename = options.potentialFileName
    if not basename:
        basename = options.type1
    modname = "pot_%s.mod" % basename
    paramname = "%s.snapparam" % basename
    coeffname = "%s.snapcoeff" % basename
    # Write mod file
    try:
        lmpoutput = open(modname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " % modname + \
                "writing.")

    print >>lmpoutput, "# Definition of SNAP+ZBL potential."
    print >>lmpoutput, "variable zblcutinner equal %g" % options.zblcutinner
    print >>lmpoutput, "variable zblcutouter equal %g" % options.zblcutouter
    print >>lmpoutput, "variable zblz equal %g" % options.zblz1

    print >>lmpoutput, \
        """
# Specify hybrid with SNAP, ZBL, and long-range Coulomb

pair_style hybrid/overlay &
zbl ${zblcutinner} ${zblcutouter} &
snap
pair_coeff 1 1 zbl ${zblz} ${zblz}
pair_coeff * * snap %s.snapcoeff %s.snapparam %s
""" % (basename, basename, options.type1)

    lmpoutput.close()
    # Write SNAP params file
    try:
        lmpoutput = open(paramname,'w')
    except:
        raise LDECKException("Error: Could not open %s for " %paramname + \
                "writing.")
    print >>lmpoutput, "# required"
    print >>lmpoutput, "rcutfac %g" % options.rcutfac
    print >>lmpoutput, "twojmax %g" % options.twojmax
    print >>lmpoutput, "\n# optional\n"
#    print >>lmpoutput, "gamma %g" % options.gamma
    print >>lmpoutput, "rfac0 %g" % options.rfac0
    print >>lmpoutput, "rmin0 %g" % options.rmin0
    print >>lmpoutput, "bzeroflag %d"  % options.bzeroflag
    print >>lmpoutput, "quadraticflag %d \n"  % options.quadratic
    lmpoutput.close()

    # Write SNAP coefficients file
    fp = open(coeffname,"w")
    fp.write("# LAMMPS SNAP coefficients for %s\n\n" % options.type1)
    fp.write("1 %d\n" % (numCoeffs+1,))
    fp.write("%s %g %g\n" %(options.type1,options.radelem1, 1.0))
#    if options.numConstants < options.numTypes:
#        fp.write("  0.000000000000\n")
    if options.quadratic == 0:
        line=0
        for coeff in SNAPCoeff[:numCoeffs+1]:
            fp.write("%16.12f # B%s \n"%(coeff, coeffindices[line]))
            #fp.write("%16.12f\n" % coeff)
            line+=1
        fp.close()
    else:
        for coeff in SNAPCoeff[:numCoeffs+1]:
            fp.write("%16.12f\n" % coeff)
        fp.close()

def gen_lammps_script(SNAPCoeff,numCoeffs,coeffindices):
    print "Generating LAMMPS run script."

    if options.numTypes == 1:
        _one_species(SNAPCoeff,numCoeffs,coeffindices)
    elif options.numTypes == 2:
        _two_species(SNAPCoeff,numCoeffs,coeffindices)
    elif options.numTypes == 3:
        _three_species(SNAPCoeff,numCoeffs,coeffindices)
    else:
        _four_species(SNAPCoeff,numCoeffs,coeffindices)
    return
