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
import numpy as np
from lammps import lammps
from random import random
try:
    # stubs = 0 MPI is active
    stubs = 0
    from mpi4py import MPI
except ModuleNotFoundError:
    stubs = 1


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


def stub_check(method):
    def stub_function(*args, **kw):
        if stubs == 0:
            return method(*args, **kw)
        else:
            return dummy_function(*args, **kw)
    return stub_function


def print_lammps(method):
    def new_method(*args, **kw):
        print(*args)
        return method(*args, **kw)
    return new_method


class ParallelTools:

    def __init__(self):
        if stubs == 0:
            self._comm = MPI.COMM_WORLD
            self._rank = self._comm.Get_rank()
            self._size = self._comm.Get_size()
        if stubs == 1:
            self._rank = 0
            self._size = 1
            self._comm = None
            self._sub_rank = 0
            self._sub_size = 1
            self._sub_comm = None
            self._sub_head_proc = 0
            self._node_index = 0
            self._number_of_nodes = 1

        self._comm_split()
        self._lmp = None
        self._seed = 0.0
        self._set_seed()
        self.shared_arrays = {}

    @stub_check
    def _comm_split(self):
        self._sub_comm = self._comm.Split_type(MPI.COMM_TYPE_SHARED)
        self._sub_rank = self._sub_comm.Get_rank()
        self._sub_size = self._sub_comm.Get_size()
        self._sub_head_proc = 0
        if self._sub_rank == 0:
            self._sub_head_proc = self._rank
        self._sub_head_proc = self._sub_comm.bcast(self._sub_head_proc)
        self._sub_head_procs = list(dict.fromkeys(self._comm.allgather(self._sub_head_proc)))
        self._head_group = self._comm.group.Incl(self._sub_head_procs)
        self._head_group_comm = self._comm.Create_group(self._head_group)
        self._node_index = self._sub_head_procs.index(self._sub_head_proc)
        self._number_of_nodes = len(self._sub_head_procs)
        self._micro_comm = self._comm.Split(self._rank)

    def _set_seed(self):
        if self._rank == 0.0:
            self._seed = random()
        if stubs == 0:
            self._seed = self._comm.bcast(self._seed)

    def get_seed(self):
        return self._seed

    def get_rank(self):
        return self._rank

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

    def rank_zero(self, method):
        if self._rank == 0:
            def check_if_rank_zero(*args, **kw):
                return method(*args, **kw)
            return check_if_rank_zero
        else:
            return dummy_function

    def sub_rank_zero(self, method):
        if self._sub_rank == 0:
            def check_if_rank_zero(*args, **kw):
                return method(*args, **kw)
            return check_if_rank_zero
        else:
            return dummy_function

    def create_shared_array(self, name, size1, size2=1, dtype='d'):

        if isinstance(name, str):
            if stubs == 0:
                self.shared_arrays[name] = SharedArray(size1, size2=size2,
                                                       dtype=dtype,
                                                       comm=self._sub_comm,
                                                       rank=self._sub_rank)
            else:
                self.shared_arrays[name] = StubsArray(size1, size2, dtype=dtype)
        else:
            raise TypeError("name must be a string")

    @stub_check
    def all_barrier(self):
        self._comm.Barrier()

    @stub_check
    def sub_barrier(self):
        self._sub_comm.Barrier()

    def split_by_node(self, obj):
        if isinstance(obj, list):
            return obj[self._node_index::self._number_of_nodes]
        elif isinstance(obj, dict):
            for key in obj:
                obj[key] = obj[key][self._node_index::self._number_of_nodes]
            return obj
        else:
            raise TypeError("Parallel tools cannot split {} by node.".format(obj))

    def split_within_node(self, obj):
        if isinstance(obj, list):
            return obj[self._sub_rank::self._sub_size]
        elif isinstance(obj, dict):
            for key in obj:
                obj[key] = obj[key][self._node_index::self._number_of_nodes]
            return obj
        else:
            raise TypeError("Parallel tools cannot split {} within node.".format(obj))

    def initialize_lammps(self, lammpslog=0, printlammps=0):
        cmds = ["-screen", "none"]
        if not lammpslog:
            cmds.append("-log")
            cmds.append("none")
        if stubs == 0:
            self._lmp = lammps(comm=self._micro_comm, cmdargs=cmds)
        else:
            self._lmp = lammps(cmdargs=cmds)

        if printlammps == 1:
            self._lmp.command = print_lammps(self._lmp.command)
        return self._lmp

    def close_lammps(self):
        if self._lmp is not None:
            # Kill lammps jobs
            self._lmp.close()
            self._lmp = None
        return self._lmp

    def slice_array(self, name, num_types=None):
        if name in self.shared_arrays:
            if name != 'a':
                s = slice(self._sub_rank, None, self._sub_size)
                self.shared_arrays[name].sliced_array = self.shared_arrays[name].array[s][:]
            else:
                self.slice_a()
        else:
            raise IndexError("{} not found in shared objects".format(name))

    def slice_a(self, name='a'):
        nof = len(pt.shared_arrays["number_of_atoms"].array)
        s = slice(self._sub_rank, nof, self._sub_size)
        self.shared_arrays[name].energies_index = self.shared_arrays[name].array[s][:]
        e_temp = []
        for i in range(nof):
            e_temp.append(i)
        f_temp = [e_temp[-1] + 1]
        s_temp = []
        for i in range(nof):
            noa = pt.shared_arrays["number_of_atoms"].array[i]
            f_temp.append(3 * noa + f_temp[-1])
        for i in range(nof):
            s_temp.append(6 * i + f_temp[-1])
        self.shared_arrays[name].energies_index = e_temp[s]
        self.shared_arrays[name].forces_index = f_temp[s]
        self.shared_arrays[name].stress_index = s_temp[s]

    @stub_check
    def combine_coeffs(self, coeff):
        new_coeff = None
        if self._sub_rank == 0:
            new_coeff = self._head_group_comm.allreduce(coeff)/self._number_of_nodes
        return self._sub_comm.bcast(new_coeff)


class SharedArray:

    def __init__(self, size1, size2=1, dtype='d', comm=None, rank=0):

        # total array for all procs
        self.array = None
        # sub array for this proc
        self.sliced_array = None

        self.energies_index = None
        self.forces_index = None
        self.strain_index = None

        if dtype == 'd':
            item_size = MPI.DOUBLE.Get_size()
        elif dtype == 'i':
            item_size = MPI.INT.Get_size()
        else:
            raise TypeError("dtype {} has not been implemented yet".format(dtype))
        if rank == 0:
            nbytes = size1 * size2 * item_size
        else:
            nbytes = 0

        win = MPI.Win.Allocate_shared(nbytes, item_size, comm=comm)

        buff, item_size = win.Shared_query(0)

        if dtype == 'd':
            assert item_size == MPI.DOUBLE.Get_size()
        elif dtype == 'i':
            assert item_size == MPI.INT32_T.Get_size()
        if size2 == 1:
            self.array = np.ndarray(buffer=buff, dtype=dtype, shape=(size1, ))
        else:
            self.array = np.ndarray(buffer=buff, dtype=dtype, shape=(size1, size2))


class StubsArray:

    def __init__(self, size1, size2=1, dtype='d'):
        # total array for all procs
        self.array = None
        # sub array for this proc
        self.sliced_array = None

        self.energies_index = None
        self.forces_index = None
        self.strain_index = None

        if size2 == 1:
            self.array = np.ndarray(shape=(size1, ), dtype=dtype)
        else:
            self.array = np.ndarray(shape=(size1, size2), dtype=dtype)


if __name__ == "fitsnap3.parallel_tools":
    pt = ParallelTools()
