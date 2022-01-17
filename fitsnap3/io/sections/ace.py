import numpy as np
from os import path
from copy import deepcopy
from time import time

from .sections import Section
from ...lib.pace.pace import CouplingCoeffs, intermediates
from ...parallel_tools import pt


class Ace(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)

        self.allowedkeys = ['numTypes', 'ranks', 'lmax', 'nmax', 'nmaxbase', 'rcutfac',
                            'lambda', 'type', 'bzeroflag', 'wigner_flag', 'acefile', 'dorder']
        self._check_section()

        self.lmbda = self.get_value("ACE", "lambda", '5.0', "float")
        self.numtypes = self.get_value("ACE", "numTypes", "1", "int")
        self.ranks = self.get_value("ACE", "ranks", "3").split()
        self.lmax = self.get_value("ACE", "lmax", "2").split()
        self.nmax = self.get_value("ACE", "nmax", "2").split()
        self.zipped = zip(self.ranks, self.lmax, self.nmax)
        self.nmaxbase = self.get_value("ACE", "nmaxbase", "16", "int")
        self.rcutfac = self.get_value("ACE", "rcutfac", "7.5", "float")
        self.types = self.get_value("ACE", "type", "H").split()
        self.type_mapping = {}
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        self.bzeroflag = self.get_value("ACE", "bzeroflag", "0", "bool")
        self.wigner_flag = self.get_value("ACE", "wigner_flag", "1", "bool")
        self.acefile = self.get_value("ACE", "acefile", "coupling_coefficients.ace")
        self.dorder = self.get_value("ACE", "dorder", "1", "bool")
        self.ranked_nus = [GenerateNL(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), self.dorder) for ind, rnk in
                           enumerate(self.ranks)]
        self.nus = [item for sublist in self.ranked_nus for item in sublist.nl]
        self._generate_b_list()
        self._write_couple()
        self.delete()

    def _generate_b_list(self):
        self.blist = []
        self.blank2J = []
        prefac = 1.0
        i = 0

        for atype in range(self.numtypes):
            for nu in self.nus:
                i += 1
                self.blist.append([i] + [int(k) for k in nu.split(',')])
                self.blank2J.append([prefac])
        self.ncoeff = int(len(self.blist))
        if not self.bzeroflag:
            self.blank2J = np.reshape(self.blank2J, (self.numtypes, int(len(self.blist)/self.numtypes)))
            onehot_atoms = np.ones((self.numtypes, 1))
            self.blank2J = np.concatenate((onehot_atoms, self.blank2J), axis=1)
            self.blank2J = np.reshape(self.blank2J, (len(self.blist) + self.numtypes))
        else:
            self.blank2J = np.reshape(self.blank2J, len(self.blist))

    def _write_couple(self):
        if not path.isfile(self.acefile):
            ccs = CouplingCoeffs(self.ranked_nus, wflag=self.wigner_flag)
            # TODO figure out where E0 comes from in Fitsnap
            ccs.write_pot(self.acefile, self.types[0], nradbase=self.nmaxbase, rcut=self.rcutfac, exp_lambda=self.lmbda)
        else:
            pt.single_print('USING EXISTING coupling_coefficients.ace -this file will need '
                            'to be deleted if potential parameters are changed in the FitSNAP infile!')


class GenerateNL:

    def __init__(self, rank, nmax, lmax, enforce_dorder=True):
        self.rank = rank
        self.nmax = nmax
        self.lmax = lmax
        self.enforce_dorder = enforce_dorder
        self._llist = []
        self.nl = []
        self._this_nl = None
        self.nl_dict = {}
        self._larray = None
        self._marray = None
        self._narray = None
        self._indices = None
        self._w3j = None
        self._generate_nl()
        del self._llist
        del self._larray
        del self._narray

    def _generate_nl(self):
        if self.rank == 1:
            for n in range(1, self.nmax + 1):
                self.nl.append('%d,0,0' % n)
            return
        if self.rank == 2:
            for n1 in range(1, self.nmax + 1):
                for n2 in range(1, self.nmax + 1):
                    for l in range(self.lmax + 1):
                        x = [(l, n1), (l, n2)]
                        srt = sorted(x)
                        if x == srt:
                            self.nl.append('%d,%d,%d' % (n1, n2, l))
            return
        self._larray = np.zeros(2 * (self.rank - 2) + 1, dtype=np.int)
        self._narray = np.zeros(self.rank, dtype=np.int)
        self._generate_l()
        loop_count = self.rank
        for i in range(1, self.nmax + 1):
            self._narray[0] = i
            self._nloop(loop_count)

    def _generate_l(self):
        loop_count = 2 * (self.rank - 2)
        for i in range(self.lmax + 1):
            self._larray[-(loop_count + 1)] = i
            self._lloop(loop_count)
        self._llist = np.array(self._llist)
        indices = np.zeros(np.shape(self._llist[:, 1::2])[-1] + 2, dtype=np.int)
        for ind in range(len(indices) - 2):
            indices[ind + 1] = 1 + ind * 2
        indices[-1] = indices[-2] + 1
        self._llist = self._llist[:, indices]
        self._llist = self._llist.tolist()

    def _lloop(self, loop_count):
        loop_count -= 1
        if loop_count == 0:
            for i in intermediates(self._larray[-2], self._larray[-3]):
                self._larray[-1] = i
                # check triplet sums are evens
                trip_checks = int(((len(self._larray) - 1) / 2) - 1)
                flag = 0
                for check in range(trip_checks):
                    val = check * 2
                    if np.sum(self._larray[val:val + 3]) % 2 != 0:
                        flag += 1
                        break
                # check final value greater than lmax and if sum of l's are even
                if flag or i > self.lmax or (np.sum(self._larray[1::2]) + self._larray[0] + self._larray[-1]) % 2 != 0:
                    continue
                self._llist.append(deepcopy(self._larray))
            return
        elif loop_count % 2 == 0:
            for i in intermediates(self._larray[-(loop_count + 2)], self._larray[-(loop_count + 3)]):
                self._larray[-(loop_count + 1)] = i
                self._lloop(loop_count)
        else:
            for i in range(self.lmax + 1):
                self._larray[-(loop_count + 1)] = i
                self._lloop(loop_count)

    def _nloop(self, loop_count):
        loop_count -= 1
        if loop_count == 0:
            for ls in self._llist:
                x = [(ls[i], self._narray[i]) for i in range(self.rank)]
                srt = sorted(x)
                if x == srt:
                    nstr = ""
                    lstr = ""
                    if self.enforce_dorder:
                        for i in range(self.rank):
                            nstr += "%d," % self._narray[-(i + 1)]
                            lstr += "%d," % ls[-(i + 1)]
                        stmp = (nstr + lstr).rstrip(',')
                        if stmp not in self.nl:
                            self.nl.append(stmp)
                    else:
                        for i in range(self.rank):
                            nstr += "%d," % self._narray[i]
                            lstr += "%d," % ls[i]
                        stmp = (nstr + lstr).rstrip(',')
                        if stmp not in self.nl:
                            self.nl.append(stmp)
            return
        for i in range(1, self.nmax + 1):
            self._narray[-loop_count] = i
            self._nloop(loop_count)

    def ccs(self, w3j):
        start = time()
        pt.single_print("Calculating Wigner Coupling for Rank {}".format(self.rank))
        self._w3j = w3j
        if self.rank == 1:
            for nl in self.nl:
                self.nl_dict[nl] = {'0': 1.}

        elif self.rank == 2:
            for nl in self.nl:
                self.nl_dict[nl] = {}
                l1 = np.fromstring(nl, dtype=int, sep=',')[-1]
                for m in range(-l1, l1 + 1):
                    self.nl_dict[nl]['%d,%d' % (m, -m)] = ((-1) ** m)

        else:
            self._larray = np.zeros(2 * (self.rank - 2) + 1, dtype=np.int)
            self._marray = np.zeros(2 * (self.rank - 2) + 1, dtype=np.int)
            self._indices = [0]
            self._indices.extend([*range(1, len(self._larray)-1, 2)])
            self._indices.append(-1)
            for i, nl in enumerate(self.nl):
                pt.single_print("rank {} {:2.2f}% done".format(self.rank, 100*i/len(self.nl)), overwrite=True)
                self._this_nl = nl
                self._larray[self._indices] = np.fromstring(nl, dtype=int, sep=',')[-self.rank:]
                self.nl_dict[nl] = {}
                self._tri_loop(self.rank-2)

        pt.single_print("\nRank {} took {} ms".format(self.rank, time()-start))

    def _tri_loop(self, loop_count):
        loop_count -= 1
        if loop_count == 0:
            for m in range(-self._larray[0], self._larray[0] + 1):
                self._marray[0] = m
                self._m_loop(self.rank-2)
            return
        for i in intermediates(self._larray[-loop_count*2-3], self._larray[-loop_count*2-2]):
            self._larray[-loop_count*2-1] = i
            self._tri_loop(loop_count)

    def _m_loop(self, loop_count):
        loop_count -= 1
        if loop_count == 0:
            self._final_mloop()
            return
        for m in range(-self._larray[-loop_count*2-2], self._larray[-loop_count*2-2] + 1):
            for mt in range(-self._larray[-loop_count*2-1], self._larray[-loop_count*2-1] + 1):
                if (self._marray[-loop_count*2-3]+m) == mt:
                    self._marray[-loop_count*2-2] = m
                    self._marray[-loop_count*2-1] = mt
                    self._m_loop(loop_count)

    def _final_mloop(self):
        for mn2 in range(-self._larray[-2], self._larray[-2] + 1):
            for mn1 in range(-self._larray[-1], self._larray[-1] + 1):
                self._marray[-2] = mn2
                self._marray[-1] = mn1
                if np.sum(self._marray[self._indices]) == 0:
                    mlst = ['%d']*len(self._indices)
                    mstr = ','.join(str(i) for i in mlst) % tuple(self._marray[self._indices])
                    w = 1
                    for i in range(self.rank-3):
                        w *= self._w3j['%d,%d,%d,%d,%d,%d' % (self._larray[i*2], self._marray[i*2],
                                                              self._larray[i*2+1], self._marray[i*2+1],
                                                              self._larray[i*2+2], -self._marray[i*2+2])]
                    w *= self._w3j['%d,%d,%d,%d,%d,%d' % (self._larray[-2], self._marray[-2],
                                                          self._larray[-1], self._marray[-1],
                                                          self._larray[-3], self._marray[-3])]
                    w *= ((-1) ** (np.abs(np.sum(self._marray[2:-2][::2]))))
                    try:
                        # pt.single_print(self.rank, mstr)
                        self.nl_dict[self._this_nl][mstr] += w
                    except KeyError:
                        self.nl_dict[self._this_nl][mstr] = w
        return

