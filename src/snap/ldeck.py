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

class LDECKException(SNAPException):
    pass

def _four_species(blist):
    try:
        lmpoutput = open("lmpoutput.inc",'w')
    except:
        raise LDECKException("Error: Could not open lmpoutput.inc for " + \
                "writing.")
    if (options.PairInclude != 'default_pair.inc'):
        try:
            lmppair = open(options.PairInclude,'r')
        except:
            raise LDECKException("Error: Could not open %s for reading." % options.PairInclude)
    else:
        lmppair = open("default_pair.inc",'w')

    for itype in range(1,options.numTypes+1):
        print >>lmpoutput, "group snapgroup%1d type %1d" % (itype,itype)

    ncoeff = len(blist)

    print >>lmpoutput, "variable twojmax equal %d" % options.twojmax
    print >>lmpoutput, "variable rcutfac equal %g" % options.rcutfac
    print >>lmpoutput, "variable rfac0 equal %g" % options.rfac0
    print >>lmpoutput, "variable rmin0 equal %g" % options.rmin0
    print >>lmpoutput, "variable wj1 equal %g" % options.wj1
    print >>lmpoutput, "variable wj2 equal %g" % options.wj2
    print >>lmpoutput, "variable wj3 equal %g" % options.wj3
    print >>lmpoutput, "variable wj4 equal %g" % options.wj4
    print >>lmpoutput, "variable radelem1 equal %g" % options.radelem1
    print >>lmpoutput, "variable radelem2 equal %g" % options.radelem2
    print >>lmpoutput, "variable radelem3 equal %g" % options.radelem3
    print >>lmpoutput, "variable radelem4 equal %g" % options.radelem4
    print >>lmpoutput, "variable bzero equal %g \n" % options.bzeroflag
    #    print >>lmpoutput, "variable DumpPath equal %g" % options.dumpPath
    print >>lmpoutput, "variable quad equal %g" % options.quadratic
    if (options.PairInclude == 'default_pair.inc'):
        print >>lmppair, "variable zblcutinner equal %g" % options.zblcutinner
        print >>lmppair, "variable zblcutouter equal %g" % options.zblcutouter
        print >>lmppair, "variable zblz1 equal %g" % options.zblz1
        print >>lmppair, "variable zblz2 equal %g" % options.zblz2
        print >>lmppair, "variable zblz3 equal %g" % options.zblz3
        print >>lmppair, "variable zblz4 equal %g" % options.zblz4
        if(options.qcoul > 0.0):
            print >>lmppair,"set type %1d charge %g" % (1,options.qcoul)
            print >>lmppair,"set type %1d charge %g" % (2,-options.qcoul)
            print >>lmppair,"set type %1d charge %g" % (3,options.qcoul)
            print >>lmppair,"set type %1d charge %g" % (4,-options.qcoul)
            print >>lmppair, "variable rcoul equal %g" % options.rcoul
            print >>lmppair, \
            """
            pair_style hybrid/overlay coul/long ${rcoul} zbl ${zblcutinner} ${zblcutouter}
            kspace_style ewald 1.0e-5
            pair_coeff * * coul/long
            """
        else:
            print >>lmppair, \
            """
            pair_style hybrid/overlay lj/cut ${rcutfac} zbl ${zblcutinner} ${zblcutouter}
            pair_coeff * * lj/cut 0.0 1.0
            """
        print >>lmppair, \
        """
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
        """
    print >>lmpoutput, "include %s " % options.PairInclude

    print >>lmpoutput, \
        """
# Bispectrum coefficient computes

compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3} ${radelem4} ${wj1} ${wj2} ${wj3} ${wj4} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute db all snad/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3} ${radelem4} ${wj1} ${wj2} ${wj3} ${wj4} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute vb all snav/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3}  ${radelem4} ${wj1} ${wj2} ${wj3} ${wj4} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute		bsum1 snapgroup1 reduce sum c_b[*]
compute		bsum2 snapgroup2 reduce sum c_b[*]
compute		bsum3 snapgroup3 reduce sum c_b[*]
compute		bsum4 snapgroup4 reduce sum c_b[*]
compute		vbsum all reduce sum c_vb[*]

# Print b and vb in thermo output

thermo 100
thermo_style	custom step temp ke pe c_sume etotal vol pxx pyy pzz pyz pxz pxy c_bsum1[*] c_bsum2[*] c_bsum3[*] c_bsum4[*] c_vbsum[*]
thermo_modify format float %20.15g"""

    print >>lmpoutput,\
        """
# This dumps the forces, energies, and bispectrum coefficients
dump mydump all custom 1000 ${DumpPath}/dump_${i} id type x y z fx fy fz c_e c_b[*]
dump_modify mydump sort id format float %20.15g
dump mydump_db all custom 1000 ${DumpPath}/dump_db_${i} c_db[*]
dump_modify mydump_db sort id format float %20.15g
 """

def _three_species(blist):
    try:
        lmpoutput = open("lmpoutput.inc",'w')
    except:
        raise LDECKException("Error: Could not open lmpoutput.inc for " + \
                "writing.")
    if (options.PairInclude != 'default_pair.inc'):
        try:
            lmppair = open(options.PairInclude,'r')
        except:
            raise LDECKException("Error: Could not open %s for reading." % options.PairInclude)
    else:
        lmppair = open("default_pair.inc",'w')

    for itype in range(1,options.numTypes+1):
        print >>lmpoutput, "group snapgroup%1d type %1d" % (itype,itype)

    ncoeff = len(blist)

    print >>lmpoutput, "variable twojmax equal %d" % options.twojmax
    print >>lmpoutput, "variable rcutfac equal %g" % options.rcutfac
    print >>lmpoutput, "variable rfac0 equal %g" % options.rfac0
    print >>lmpoutput, "variable rmin0 equal %g" % options.rmin0
    print >>lmpoutput, "variable wj1 equal %g" % options.wj1
    print >>lmpoutput, "variable wj2 equal %g" % options.wj2
    print >>lmpoutput, "variable wj3 equal %g" % options.wj3
    print >>lmpoutput, "variable radelem1 equal %g" % options.radelem1
    print >>lmpoutput, "variable radelem2 equal %g" % options.radelem2
    print >>lmpoutput, "variable radelem3 equal %g" % options.radelem3
    print >>lmpoutput, "variable bzero equal %g \n" % options.bzeroflag
    #    print >>lmpoutput, "variable DumpPath equal %g" % options.dumpPath
    print >>lmpoutput, "variable quad equal %g" % options.quadratic
    if (options.PairInclude == 'default_pair.inc'):
        print >>lmppair, "variable zblcutinner equal %g" % options.zblcutinner
        print >>lmppair, "variable zblcutouter equal %g" % options.zblcutouter
        print >>lmppair, "variable zblz1 equal %g" % options.zblz1
        print >>lmppair, "variable zblz2 equal %g" % options.zblz2
        print >>lmppair, "variable zblz3 equal %g" % options.zblz3
        if(options.qcoul > 0.0):
            print >>lmppair,"set type %1d charge %g" % (1,options.qcoul)
            print >>lmppair,"set type %1d charge %g" % (2,(-2*options.qcoul))
            print >>lmppair,"set type %1d charge %g" % (3,options.qcoul)
            print >>lmppair, "variable rcoul equal %g" % options.rcoul
            print >>lmppair, \
            """
            pair_style hybrid/overlay coul/long ${rcoul} zbl ${zblcutinner} ${zblcutouter}
            kspace_style ewald 1.0e-5
            pair_coeff * * coul/long
            """
        else:
            print >>lmppair, \
            """
            pair_style hybrid/overlay lj/cut ${rcutfac} zbl ${zblcutinner} ${zblcutouter}
            pair_coeff * * lj/cut 0.0 1.0
            """
        print >>lmppair, \
        """
        pair_coeff 1 1 zbl ${zblz1} ${zblz1}
        pair_coeff 1 2 zbl ${zblz1} ${zblz2}
        pair_coeff 1 3 zbl ${zblz1} ${zblz3}
        pair_coeff 2 2 zbl ${zblz2} ${zblz2}
        pair_coeff 2 3 zbl ${zblz2} ${zblz3}
        pair_coeff 3 3 zbl ${zblz3} ${zblz3}
        """
    print >>lmpoutput, "include %s " % options.PairInclude

    print >>lmpoutput, \
        """
# Bispectrum coefficient computes

compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3} ${wj1} ${wj2} ${wj3} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute db all snad/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3} ${wj1} ${wj2} ${wj3} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute vb all snav/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${radelem3} ${wj1} ${wj2} ${wj3} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute		bsum1 snapgroup1 reduce sum c_b[*]
compute		bsum2 snapgroup2 reduce sum c_b[*]
compute		bsum3 snapgroup3 reduce sum c_b[*]
compute		vbsum all reduce sum c_vb[*]

# Print b and vb in thermo output

thermo 100
thermo_style	custom step temp ke pe c_sume etotal vol pxx pyy pzz pyz pxz pxy c_bsum1[*] c_bsum2[*] c_bsum3[*] c_vbsum[*]
thermo_modify format float %20.15g"""

    print >>lmpoutput,\
        """
# This dumps the forces, energies, and bispectrum coefficients
dump mydump all custom 1000 ${DumpPath}/dump_${i} id type x y z fx fy fz c_e c_b[*]
dump_modify mydump sort id format float %20.15g
dump mydump_db all custom 1000 ${DumpPath}/dump_db_${i} c_db[*]
dump_modify mydump_db sort id format float %20.15g
 """

def _two_species(blist):
    try:
        lmpoutput = open("lmpoutput.inc",'w')
    except:
        raise LDECKException("Error: Could not open lmpoutput.inc for " + \
                "writing.")
    if (options.PairInclude != 'default_pair.inc'):
        try:
            lmppair = open(options.PairInclude,'r')
        except:
            raise LDECKException("Error: Could not open %s for reading." % options.PairInclude)
    else:
        lmppair = open("default_pair.inc",'w')

    for itype in range(1,options.numTypes+1):
        print >>lmpoutput, "group snapgroup%1d type %1d" % (itype,itype)

    ncoeff = len(blist)

    print >>lmpoutput, "variable twojmax equal %d" % options.twojmax
    print >>lmpoutput, "variable rcutfac equal %g" % options.rcutfac
    print >>lmpoutput, "variable rfac0 equal %g" % options.rfac0
    print >>lmpoutput, "variable rmin0 equal %g" % options.rmin0
    print >>lmpoutput, "variable wj1 equal %g" % options.wj1
    print >>lmpoutput, "variable wj2 equal %g" % options.wj2
    print >>lmpoutput, "variable radelem1 equal %g" % options.radelem1
    print >>lmpoutput, "variable radelem2 equal %g" % options.radelem2
    print >>lmpoutput, "variable bzero equal %g \n" % options.bzeroflag
    #    print >>lmpoutput, "variable DumpPath equal %g" % options.dumpPath
    print >>lmpoutput, "variable quad equal %g" % options.quadratic
    if (options.PairInclude == 'default_pair.inc'):
        print >>lmppair, "variable zblcutinner equal %g" % options.zblcutinner
        print >>lmppair, "variable zblcutouter equal %g" % options.zblcutouter
        print >>lmppair, "variable zblz1 equal %g" % options.zblz1
        print >>lmppair, "variable zblz2 equal %g" % options.zblz2
        if(options.qcoul > 0.0):
            print >>lmppair,"set type %1d charge %g" % (1,options.qcoul)
            print >>lmppair,"set type %1d charge %g" % (2,(-options.qcoul))
            print >>lmppair, "variable rcoul equal %g" % options.rcoul
            print >>lmppair, \
            """
            pair_style hybrid/overlay coul/long ${rcoul} zbl ${zblcutinner} ${zblcutouter}
            kspace_style ewald 1.0e-5
            pair_coeff * * coul/long
            """
        else:
            print >>lmppair, \
            """
            pair_style hybrid/overlay lj/cut ${rcutfac} zbl ${zblcutinner} ${zblcutouter}
            pair_coeff * * lj/cut 0.0 1.0
            """
        print >>lmppair, \
        """
        pair_coeff 1 1 zbl ${zblz1} ${zblz1}
        pair_coeff 1 2 zbl ${zblz1} ${zblz2}
        pair_coeff 2 2 zbl ${zblz2} ${zblz2}
        """
    print >>lmpoutput, "include %s " % options.PairInclude

    print >>lmpoutput, \
        """
# Bispectrum coefficient computes

compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${wj1} ${wj2} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute db all snad/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${wj1} ${wj2} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute vb all snav/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${radelem2} ${wj1} ${wj2} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute		bsum1 snapgroup1 reduce sum c_b[*]
compute		bsum2 snapgroup2 reduce sum c_b[*]
compute		vbsum all reduce sum c_vb[*]

# Print b and vb in thermo output

thermo 100
thermo_style	custom step temp ke pe c_sume etotal vol pxx pyy pzz pyz pxz pxy c_bsum1[*] c_bsum2[*] c_vbsum[*]
thermo_modify format float %20.15g"""

    print >>lmpoutput,\
        """
# This dumps the forces, energies, and bispectrum coefficients
dump mydump all custom 1000 ${DumpPath}/dump_${i} id type x y z fx fy fz c_e c_b[*]
dump_modify mydump sort id format float %20.15g
dump mydump_db all custom 1000 ${DumpPath}/dump_db_${i} c_db[*]
dump_modify mydump_db sort id format float %20.15g
 """

def _one_species(blist):
    try:
        lmpoutput = open("lmpoutput.inc",'w')
    except:
        raise LDECKException("Error: Could not open lmpoutput.inc for " + \
                "writing.")
    if (options.PairInclude != 'default_pair.inc'):
        try:
            lmppair = open(options.PairInclude,'r')
        except:
            raise LDECKException("Error: Could not open %s for reading." % options.PairInclude)
    else:
        lmppair = open("default_pair.inc",'w')

    for itype in range(1,options.numTypes+1):
        print >>lmpoutput, "group snapgroup%1d type %1d" % (itype,itype)

    ncoeff = len(blist)

    print >>lmpoutput, "variable twojmax equal %d" % options.twojmax
    print >>lmpoutput, "variable rcutfac equal %g" % options.rcutfac
    print >>lmpoutput, "variable rfac0 equal %g" % options.rfac0
    print >>lmpoutput, "variable rmin0 equal %g" % options.rmin0
    print >>lmpoutput, "variable wj1 equal %g" % options.wj1
    print >>lmpoutput, "variable radelem1 equal %g" % options.radelem1
    print >>lmpoutput, "variable bzero equal %g \n" % options.bzeroflag
    #    print >>lmpoutput, "variable DumpPath equal %g" % options.dumpPath
    print >>lmpoutput, "variable quad equal %g" % options.quadratic
    if (options.PairInclude == 'default_pair.inc'):
        print >>lmppair, "variable zblcutinner equal %g" % options.zblcutinner
        print >>lmppair, "variable zblcutouter equal %g" % options.zblcutouter
        print >>lmppair, "variable zblz1 equal %g" % options.zblz1
        if(options.qcoul > 0.0):
            print >>lmppair,"set type %1d charge %g" % (1,(options.qcoul))
            print >>lmppair, "variable rcoul equal %g" % options.rcoul
            print >>lmppair, \
            """
            pair_style hybrid/overlay coul/long ${rcoul} zbl ${zblcutinner} ${zblcutouter}
            kspace_style ewald 1.0e-5
            pair_coeff * * coul/long
            """
        else:
            print >>lmppair, \
            """
            pair_style hybrid/overlay lj/cut ${rcutfac} zbl ${zblcutinner} ${zblcutouter}
            pair_coeff * * lj/cut 0.0 1.0
            """
        print >>lmppair, \
        """
        pair_coeff 1 1 zbl ${zblz1} ${zblz1}
        """
    print >>lmpoutput, "include %s " % options.PairInclude

    print >>lmpoutput, \
        """
# Bispectrum coefficient computes

compute b all sna/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${wj1} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute db all snad/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${wj1} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute vb all snav/atom ${rcutfac} ${rfac0} ${twojmax} ${radelem1} ${wj1} rmin0 ${rmin0} bzeroflag ${bzero} quadraticflag ${quad}
compute		bsum1 snapgroup1 reduce sum c_b[*]
compute		vbsum all reduce sum c_vb[*]

# Print b and vb in thermo output

thermo 100
thermo_style	custom step temp ke pe c_sume etotal vol pxx pyy pzz pyz pxz pxy c_bsum1[*] c_vbsum[*]
thermo_modify format float %20.15g"""

    print >>lmpoutput,\
        """
# This dumps the forces, energies, and bispectrum coefficients
dump mydump all custom 1000 ${DumpPath}/dump_${i} id type x y z fx fy fz c_e c_b[*]
dump_modify mydump sort id format float %20.15g
dump mydump_db all custom 1000 ${DumpPath}/dump_db_${i} c_db[*]
dump_modify mydump_db sort id format float %20.15g
 """

def _generateblist(twojmax):
    blist = []
    i = 0
#    blist.append([0,0,0,0])
    for j1 in range(twojmax+1):
        for j2 in range(j1+1):
            for j in range(abs(j1-j2), min(twojmax,j1+j2)+1,2):
                if j >= j1:
                    i += 1
                    blist.append([i,j1,j2,j])
    return blist

def _generatecoeffindices(twojmax):
    blist = []
    i = 0
    blist.append([0])
    for j1 in range(twojmax+1):
        for j2 in range(j1+1):
            for j in range(abs(j1-j2), min(twojmax,j1+j2)+1,2):
                if j >= j1:
                    i += 1
                    blist.append([i,j1,j2,j])
    return blist

def gen_lammps_script():
    lmpinput = open("in.snap",'w')
    print >>lmpinput,\
        """
shell mkdir ${DumpPath}
label loop
variable i uloop ${nfiles} pad
log ${DumpPath}/log_${i}
units           metal
atom_style      %s
atom_modify map array sort 0 2.0
box tilt large
read_data ${DataPath}/data.lammps_${i}
mass * 1.0e20
compute e all pe/atom
compute sume all reduce sum c_e
include lmpoutput.inc
neighbor 1.0e-20 nsq
neigh_modify one 10000
timestep 0.001
run            0
clear
next i
jump SELF loop
""" % ((options.atomstyle).lower())
    bispectrumList = _generateblist(options.twojmax)

    if options.numTypes == 1:
        _one_species(bispectrumList)
    elif options.numTypes == 2:
        _two_species(bispectrumList)
    elif options.numTypes == 3:
        _three_species(bispectrumList)
    else:
        _four_species(bispectrumList)
    return len(bispectrumList)
