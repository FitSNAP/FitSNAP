from fitsnap3.io.sections.sections import Section
from itertools import combinations_with_replacement
import numpy as np


class Bispectrum(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.numtypes = int(self._config.get("BISPECTRUM", "numTypes", fallback='1'))
        self.twojmax = int(self._config.get("BISPECTRUM", "twojmax", fallback='6'))
        self.rcutfac = float(self._config.get("BISPECTRUM", "rcutfac", fallback='4.67637'))
        self.rfac0 = float(self._config.get("BISPECTRUM", "rfac0", fallback='0.99363'))
        self.rmin0 = float(self._config.get("BISPECTRUM", "rmin0", fallback='0.0'))
        self.wj = []
        for i in range(self.numtypes):
            self.wj.append(float(self._config.get("BISPECTRUM", "wj{}".format(i + 1), fallback='1.0')))
        self.radelem = []
        for i in range(self.numtypes):
            self.radelem.append(float(self._config.get("BISPECTRUM", "radelem{}".format(i + 1), fallback='0.5')))
        self.types = []
        for i in range(self.numtypes):
            self.types.append(self._config.get("BISPECTRUM", "type{}".format(i+1), fallback='H'))
        self.type_mapping = {}
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        self.boltz = float(self._config.get("BISPECTRUM", "BOLTZT", fallback='10000'))
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
        if self._config.get("MODEL", "alloyflag", fallback='0') != '0':
            self.blist *= self.numtypes ** 3
            self.blist = np.array(self.blist).tolist()
        if self._config.get("MODEL", "quadraticflag", fallback='0') != '0':
            # Note, combinations_with_replacement precisely generates the upper-diagonal entries we want
            self.blist += [[i, a, b] for i, (a, b) in
                           enumerate(combinations_with_replacement(self.blist, r=2), start=len(self.blist))]
        self.ncoeff = len(self.blist)
