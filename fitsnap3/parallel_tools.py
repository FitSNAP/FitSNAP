# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
# This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Maybe me ¯\_(ツ)_/¯
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

from time import time
from mpi4py import MPI
import numpy as np


def _rank_zero(method):
    def check_if_rank_zero(*args, **kw):
        if args[0].get_rank() == 0:
            return method(*args, **kw)
        else:
            return dummy_function()
    return check_if_rank_zero


def _sub_rank_zero(method):
    def check_if_rank_zero(*args, **kw):
        if args[0].get_rank() == 0:
            return method(*args, **kw)
        else:
            return dummy_function()
    return check_if_rank_zero


def identity_decorator(self, obj):
    return obj


def _rank_zero_decorator(decorator):
    def check_if_rank_zero(*args, **kw):
        if args[0].get_rank() == 0:
            return decorator(*args, **kw)
        else:
            return identity_decorator(*args, **kw)
    return check_if_rank_zero


def dummy_function(*args, **kw):
    return None


class ParallelTools:

    def __init__(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self._comm_split()

    def _comm_split(self):
        self._sub_comm = self._comm.Split_type(0)
        self._sub_rank = self._sub_comm.Get_rank()
        self._sub_size = self._sub_comm.Get_size()

    def get_rank(self):
        return self._rank

    def get_sub_rank(self):
        return self._sub_rank

    @_rank_zero
    def single_print(self, *args, **kw):
        print(*args)

    @_sub_rank_zero
    def sub_print(self, *args, **kw):
        print(*args)

    def all_print(self, *args, **kw):
        print("Rank", self._rank, ":", *args)

    @_rank_zero_decorator
    def single_timeit(self, method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print("'{0}' took {1:.2f} ms on rank {2}".format(method.__name__, (te - ts) * 1000, self._rank))
            return result
        return timed

    def per_rank_timeit(self, method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print("'{0}' took {1:.2f} ms on rank {2}".format(method.__name__, (te - ts) * 1000, self._rank))
            return result
        return timed

    def sub_rank_zero(self, method):
        if self._sub_rank == 0:
            def check_if_rank_zero(*args, **kw):
                return method(*args, **kw)
            return check_if_rank_zero
        else:
            return dummy_function

    def create_shared_array(self, size):
        item_size = MPI.DOUBLE.Get_size()
        if self._sub_rank == 0:
            nbytes = size * item_size
        else:
            nbytes = 0

        win = MPI.Win.Allocate_shared(nbytes, item_size, comm=self._sub_comm)

        buff, item_size = win.Shared_query(0)
        assert item_size == MPI.DOUBLE.Get_size()
        array = np.ndarray(buffer=buff, dtype='d', shape=(size,))

        return array


if __name__ == "parallel_tools":
    pt = ParallelTools()
