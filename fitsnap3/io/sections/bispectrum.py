from fitsnap3.io.sections.sections import Section
from itertools import combinations_with_replacement
import numpy as np


class Bispectrum(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)

        self.numtypes = self.get_value("BISPECTRUM", "numTypes", "1", "int")
        self.twojmax = self.get_value("BISPECTRUM", "twojmax", "6", "int")
        self.rcutfac = self.get_value("BISPECTRUM", "rcutfac", "4.67637", "float")
        self.rfac0 = self.get_value("BISPECTRUM", "rfac0", "0.99363", "float")
        self.rmin0 = self.get_value("BISPECTRUM", "rmin0", "0.0", "float")
        self.wj = []
        for i in range(self.numtypes):
            self.wj.append(self.get_value("BISPECTRUM", "wj{}".format(i + 1), "1.0", "float"))
        self.radelem = []
        for i in range(self.numtypes):
            self.radelem.append(self.get_value("BISPECTRUM", "radelem{}".format(i + 1), "0.5", "float"))
        self.types = []
        for i in range(self.numtypes):
            self.types.append(self.get_value("BISPECTRUM", "type{}".format(i + 1), "H"))
        self.type_mapping = {}
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        self._generate_b_list()
        self.delete()

    def _generate_b_list(self):
        self.blist = []
        i = 0
        for j1 in range(self.twojmax + 1):
            for j2 in range(j1 + 1):
                for j in range(abs(j1 - j2), min(self.twojmax, j1 + j2) + 1, 2):
                    if j >= j1:
                        i += 1
                        self.blist.append([i, j1, j2, j])
        if self.get_value("CALCULATOR", "chemflag", "0", "bool"):
            self.blist *= self.numtypes ** 3
            self.blist = np.array(self.blist).tolist()
        if self.get_value("CALCULATOR", "quadraticflag", "0", "bool"):
            # Note, combinations_with_replacement precisely generates the upper-diagonal entries we want
            self.blist += [[i, a, b] for i, (a, b) in
                           enumerate(combinations_with_replacement(self.blist, r=2), start=len(self.blist))]
        self.ncoeff = len(self.blist)
