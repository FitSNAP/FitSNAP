from fitsnap3lib.io.sections.sections import Section
from itertools import combinations_with_replacement
import numpy as np


class Bispectrum(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['numTypes', 'twojmax', 'rcutfac', 'rfac0', 'rmin0', 'wj', 'radelem', 'type',
                            'wselfallflag', 'chemflag', 'bzeroflag', 'quadraticflag', 'bnormflag', 'bikflag',
                            'switchinnerflag', 'switchflag', 'sinner', 'dinner', 'dgradflag']
        self._check_section()

        self._check_if_used("CALCULATOR", "calculator", "LAMMPSSNAP", "LAMMPSSNAP")

        self.numtypes = self.get_value("BISPECTRUM", "numTypes", "1", "int")
        self.twojmax = self.get_value("BISPECTRUM", "twojmax", "6").split()
        self.rcutfac = self.get_value("BISPECTRUM", "rcutfac", "4.67637", "float")
        self.rfac0 = self.get_value("BISPECTRUM", "rfac0", "0.99363", "float")
        self.rmin0 = self.get_value("BISPECTRUM", "rmin0", "0.0", "float")
        self.wj = self.get_value("BISPECTRUM", "wj", "1.0").split()
        self.radelem = self.get_value("BISPECTRUM", "radelem", "0.5").split()
        self.types = self.get_value("BISPECTRUM", "type", "H").split()
#        self.wj = []
#        self.radelem = []
#        self.types = []
        self.type_mapping = {}
#        for i in range(self.numtypes):
#            self.wj.append(self.get_value("BISPECTRUM", "wj{}".format(i + 1), "1.0", "float"))
#        for i in range(self.numtypes):
#            self.radelem.append(self.get_value("BISPECTRUM", "radelem{}".format(i + 1), "0.5", "float"))
#        for i in range(self.numtypes):
#            self.types.append(self.get_value("BISPECTRUM", "type{}".format(i + 1), "H"))
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        # chemflag true enables the EME model
        self.chemflag = self.get_value("BISPECTRUM", "chemflag", "0", "bool")
        self.bnormflag = self.get_value("BISPECTRUM", "bnormflag", "0", "bool")
        self.wselfallflag = self.get_value("BISPECTRUM", "wselfallflag", "0", "bool")
        self.bzeroflag = self.get_value("BISPECTRUM", "bzeroflag", "0", "bool")
        # quadraticflag true enables the quadratic model
        self.quadraticflag = self.get_value("BISPECTRUM", "quadraticflag", "0", "bool")
        if (self.chemflag and self.quadraticflag):
            raise ValueError("Quadratic chemsnap not impelemented.")
        # bikflag true enables computing of bispectrum per atom instead of sum
        self.bikflag = self.get_value("BISPECTRUM", "bikflag", "0", "bool")
        #if self.bikflag:
        #    self._assert_dependency('bikflag', "CALCULATOR", "per_atom_energy", True)
        self._generate_b_list()
        self._reset_chemflag()
        Section.num_desc = len(self.blist)
        # switchinnerflag true enables inner cutoff function
        self.switchinnerflag = self.get_value("BISPECTRUM", "switchinnerflag", "0", "bool")
        if (self.switchinnerflag):
            default_sinner = self.numtypes*"0.9 "
            default_dinner = self.numtypes*"0.1 "
            self.sinner = self.get_value("BISPECTRUM", "sinner", default_sinner[:-1], "str")
            self.dinner = self.get_value("BISPECTRUM", "dinner", default_dinner[:-1], "str")
            if ( (len(self.sinner.split()) != self.numtypes) or (len(self.dinner.split()) != self.numtypes)):
                raise ValueError("Number of sinner/dinner args must be number of types.")
        self.switchflag = self.get_value("BISPECTRUM", "switchflag", "1", "bool")
        # dgradflag true enables per-neighbor descriptor derivatives for nonlinear force Fitting
        self.dgradflag = self.get_value("BISPECTRUM", "dgradflag", "0", "bool")
        self.delete()

    def _generate_b_list(self):
        self.blist = []
        self.blank2J = []
# Save for when LAMMPS will accept multiple 2J
#        for atype in range(self.numtypes):
#            for j1 in range(int(self.twojmax[atype]) + 1):
#                for j2 in range(j1 + 1):
#                    for j in range(abs(j1 - j2), min(int(self.twojmax[atype]), j1 + j2) + 1, 2):
#                        if j >= j1:
#                            i += 1
#                            self.blist.append([i, j1, j2, j])
        for atype in range(self.numtypes):
            i = 0
            for j1 in range(int(max(self.twojmax)) + 1):
                for j2 in range(j1 + 1):
                    for j in range(abs(j1 - j2), min(int(max(self.twojmax)), j1 + j2) + 1, 2):
                        if j >= j1:
                            prefac = 0.0
                            if all(ind <= int(self.twojmax[atype]) for ind in [j1,j2,j]) :
                                prefac = 1.0
                            i += 1
                            self.blist.append([i, j1, j2, j])
                            self.blank2J.append([prefac])
            if self.quadraticflag:
                slice = int(len(self.blist)/(atype+1))
                start = slice*atype
                end = slice*(atype+1)
                for i, (a, b) in enumerate(combinations_with_replacement(self.blist[start:end], r=2), start=slice):
                    prefac = 0.0
                    quadIndex = a[1:]+b[1:]
                    if all(ind <= int(self.twojmax[atype]) for ind in quadIndex):
                        prefac = 1.0
                    self.blank2J.append([prefac])

        if self.chemflag:
            self.blist *= self.numtypes ** 3
            self.blist = np.array(self.blist).tolist()
            if int(min(self.twojmax)) != int(max(self.twojmax)):
                raise RuntimeError("Still working on the capability to do mixed 2J values per-element and explicit multi-element descriptors \n Aborting...!")
            self.blank2J *= self.numtypes ** 3
            self.blank2J = np.array(self.blank2J).tolist()
        if self.quadraticflag:
            # Note, combinations_with_replacement precisely generates the upper-diagonal entries we want
            self.blist = np.reshape(self.blist, (self.numtypes, -1, 4)).tolist()
            for atype in range(self.numtypes):
                self.blist[atype] += [[i, a, b] for i, (a, b) in
                                      enumerate(combinations_with_replacement(self.blist[atype], r=2),
                                                start=len(self.blist[atype]))]
            self.blist = [item for sublist in self.blist for item in sublist]
        self.ncoeff = int(len(self.blist)/self.numtypes)
        if not self.bzeroflag:
            self.blank2J = np.reshape(self.blank2J, (self.numtypes, int(len(self.blist)/self.numtypes)))
            onehot_atoms = np.ones((self.numtypes, 1))
            self.blank2J = np.concatenate((onehot_atoms, self.blank2J), axis=1)
            self.blank2J = np.reshape(self.blank2J, (len(self.blist) + self.numtypes))
        else:
            self.blank2J = np.reshape(self.blank2J, len(self.blist))

    def _reset_chemflag(self):
        if self.chemflag != 0:
            chemflag = "{}".format(self.numtypes)
            for element in self.type_mapping:
                element_type = self.type_mapping[element]
                chemflag += " {}".format(element_type - 1)
            self.chemflag = "{}".format(chemflag)
