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
#     Charles Sievers (Sandia National Labs)
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

from time import time, sleep
import numpy as np
from lammps import lammps
from random import randint
from psutil import virtual_memory
from itertools import chain
import ctypes
import signal
from inspect import isclass
from pkgutil import iter_modules
from importlib import import_module
from copy import deepcopy
try:
    # stubs = 0 MPI is active
    stubs = 0
    from mpi4py import MPI
except ModuleNotFoundError:
    stubs = 1


def printf(*args, **kw):
    kw['flush'] = True

    if 'overwrite' in kw:
        del kw['overwrite']
        kw['end'] = ''
        print("\r", end='')
        print(" ".join(map(str, args)), **kw)
    else:
        print(" ".join(map(str, args)), **kw)


class GracefulError(BaseException):

    def __init__(self, *args, **kwargs):
        pass


class GracefulKiller:

    def __init__(self, comm):
        self._comm = comm
        self._rank = 0
        self.already_killed = False
        if self._comm is not None:
            self._rank = self._comm.Get_rank()
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        if self._rank == 0:
            printf("attempting to exit gracefully")
        if self.already_killed:
            self._comm.Abort()
        raise GracefulError("exiting from exit code", signum, "at", frame)


def _rank_zero(method):
    def check_if_rank_zero(*args, **kw):
        if args[0].get_rank() == 0:
            return method(*args, **kw)
        else:
            return dummy_function()
    return check_if_rank_zero


def _sub_rank_zero(method):
    def check_if_rank_zero(*args, **kw):
        if args[0].get_subrank() == 0:
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
        printf(*args)
        return method(*args, **kw)
    return new_method


class ParallelTools:

    def __init__(self):
        if stubs == 0:
            try:
                self._comm = MPI.ADDED_COMM
            except NameError:
                self._comm = MPI.COMM_WORLD
            except AttributeError:
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

        self.killer = GracefulKiller(self._comm)

        self._comm_split()
        self._lmp = None
        self._seed = 0.0
        self._set_seed()
        self.shared_arrays = {}
        self.fitsnap_dict = {}
        self.logger = None
        self.pytest = False
        self._fp = None

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
            self._seed = randint(0,1e5)
        if stubs == 0:
            self._seed = self._comm.bcast(self._seed)

    def get_seed(self):
        return self._seed

    def get_size(self):
        return self._size

    def get_rank(self):
        return self._rank

    def get_subsize(self):
        return self._sub_size

    def get_subrank(self):
        return self._sub_rank

    def get_node(self):
        return self._node_index

    def get_number_of_nodes(self):
        return self._number_of_nodes

    def get_node_list(self):
        return self._sub_head_procs

    @_rank_zero
    def single_print(self, *args, **kw):
        printf(*args, file=self._fp)

    @_sub_rank_zero
    def sub_print(self, *args, **kw):
        printf("Node", self._node_index, ":", *args, file=self._fp)

    def all_print(self, *args, **kw):
        printf("Rank", self._rank, ":", *args, file=self._fp)

    def set_output(self, output_file, ns=False, ps=False):
        if ps:
            self._fp = open(output_file+'_{}'.format(self._rank), 'w')
        elif ns:
            if self._sub_rank == 0:
                self._fp = open(output_file + '_{}'.format(self._sub_rank), 'w')
        else:
            if self._rank == 0:
                self._fp = open(output_file, 'w')

    @_rank_zero_decorator
    def single_timeit(self, method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            elif self._fp is not None:
                printf("'{0}' took {1:.2f} ms on rank {2}".format(
                    method.__name__, (te - ts) * 1000, self._rank), file=self._fp)
            else:
                printf("'{0}' took {1:.2f} ms on rank {2}".format(
                    method.__name__, (te - ts) * 1000, self._rank))
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
                printf("'{0}' took {1:.2f} ms on rank {2}".format(
                    method.__name__, (te - ts) * 1000, self._rank))
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

    def create_shared_array(self, name, size1, size2=1, dtype='d', tm=0):

        if isinstance(name, str):
            if stubs == 0:
                comms = [[self._comm, self._rank, self._size],
                         [self._sub_comm, self._sub_rank, self._sub_size],
                         [self._head_group_comm, self._node_index, self._number_of_nodes]]
                self.shared_arrays[name] = SharedArray(size1, size2=size2,
                                                       dtype=dtype,
                                                       multinode=tm,
                                                       comms=comms)
            else:
                self.shared_arrays[name] = StubsArray(size1, size2, dtype=dtype)
        else:
            raise TypeError("name must be a string")

    # @stub_check
    def add_2_fitsnap(self, name, an_object):

        # TODO: Replace cluttered shared array hanging objects!
        if isinstance(name, str):
            if name in self.fitsnap_dict:
                raise NameError("name is already in dictionary")
            self.fitsnap_dict[name] = an_object
        else:
            raise TypeError("name must be a string")

    @stub_check
    def _bcast_fitsnap(self, name):

        if self._sub_rank == 0:
            if name not in self.fitsnap_dict:
                raise NameError("Dictionary element not yet in fitsnap_dictionary")
        else:
            self.fitsnap_dict[name] = None

        self.fitsnap_dict[name] = self._sub_comm.bcast(self.fitsnap_dict[name])

    @stub_check
    def gather_fitsnap(self, name):

        if name not in self.fitsnap_dict:
            raise NameError("Dictionary element not yet in fitsnap_dictionary")
        self.fitsnap_dict[name] = self._sub_comm.gather(self.fitsnap_dict[name], root=0)

    @stub_check
    @_sub_rank_zero
    def gather_to_head_node(self, array):
        return self._head_group_comm.allgather(array)

    @stub_check
    def gather_distributed_list(self, dist_list):
        gathered_list = self._sub_comm.allgather(dist_list)

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
        elif isinstance(obj, np.ndarray):
            return obj[self._node_index::self._number_of_nodes]
        elif isinstance(obj, SharedArray):
            scraped_length = obj.get_scraped_length()
            length = obj.get_node_length()
            lengths = np.zeros(self._number_of_nodes)
            difference = np.zeros(self._number_of_nodes)
            if self._sub_rank == 0:
                lengths[self._node_index] = scraped_length - length
                self._head_group_comm.Allreduce([lengths, MPI.DOUBLE], [difference, MPI.DOUBLE])
                if self._rank == 0:
                    if np.sum(difference) != 0:
                        raise ValueError(np.sum(difference), "sum of differences must be zero")
                difference = difference.astype(int)
                i, j, i_val, j_val = 0, 0, 0, 0
                while not np.all((difference == 0)):
                    for i, i_val in enumerate(difference):
                        if i_val > 0:
                            break
                    for j, j_val in enumerate(difference):
                        if j_val < 0:
                            break
                    val_min = min(i_val, -j_val)
                    if self._node_index == i:
                        # scraped length > scala length
                        self._comm.send(obj.array[scraped_length-i_val:scraped_length-(i_val-val_min)],
                                        dest=self._sub_head_procs[j],
                                        tag=11)
                    if self._node_index == j:
                        # scraped length < scala length
                        obj.array[length+j_val:length+(j_val+val_min)] = \
                            self._comm.recv(source=self._sub_head_procs[i], tag=11)
                    difference[j] += val_min
                    difference[i] -= val_min
                    self._head_group_comm.barrier()
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

    def check_lammps(self, lammps_noexceptions=0):
        cmds = ["-screen", "none", "-log", "none"]
        if stubs == 0:
            self._lmp = lammps(comm=self._micro_comm, cmdargs=cmds)
        else:
            self._lmp = lammps(cmdargs=cmds)

        if not (self._lmp.has_exceptions or lammps_noexceptions):
            raise Exception("Fitting interrupted! LAMMPS not compiled with C++ exceptions handling enabled")
        self._lmp.close()
        self._lmp = None

    def initialize_lammps(self, lammpslog=0, printlammps=0):
        cmds = ["-screen", "none"]
        if not lammpslog:
            cmds.append("-log")
            cmds.append("none")
        if stubs == 0:
            self._lmp = lammps(comm=self._micro_comm, cmdargs=cmds)
#            if 'ML-IAP' in self._lmp.installed_packages:
#                from lammps.mliap import activate_mliappy
#                activate_mliappy(self._lmp)
        else:
            self._lmp = lammps(cmdargs=cmds)
#            if 'ML-IAP' in self._lmp.installed_packages:
#                from lammps.mliap import activate_mliappy
#                activate_mliappy(self._lmp)

        if printlammps == 1:
            self._lmp.command = print_lammps(self._lmp.command)
        return self._lmp

    def close_lammps(self):
        if self._lmp is not None:
            # Kill lammps jobs
            self._lmp.close()
            self._lmp = None
        return self._lmp

    def slice_array(self, name):
        if name in self.shared_arrays:
            if name != 'a':
                s = slice(self._sub_rank, None, self._sub_size)
                self.shared_arrays[name].sliced_array = self.shared_arrays[name].array[s][:]
            else:
                raise NotImplementedError("Slice A using new_slice_a")
                # self.slice_a()
        else:
            raise IndexError("{} not found in shared objects".format(name))

    def new_slice_a(self):
        """ create array to show which sub a matrix indices belong to which proc """
        nof = len(self.shared_arrays["number_of_atoms"].array)
        if self._sub_rank != 0:
            # Wait for head proc on node to fill indices
            self._bcast_fitsnap("sub_a_size")
            self.fitsnap_dict["sub_a_size"] = int(self.fitsnap_dict["sub_a_size"][self._sub_rank])
            self._bcast_fitsnap("sub_a_indices")
            self.fitsnap_dict["sub_a_indices"] = self.fitsnap_dict["sub_a_indices"][self._sub_rank]
            return
        sub_a_sizes = np.zeros((self._sub_size, ), dtype=np.int)

        for i in range(nof):
            proc_number = i % self._sub_size
            if self.fitsnap_dict["energy"]:
                descriptor_rows = 1
                if self.fitsnap_dict["per_atom_energy"]:
                    descriptor_rows = self.shared_arrays["number_of_atoms"].array[i]
                sub_a_sizes[proc_number] += descriptor_rows
            if self.fitsnap_dict["force"]:
                sub_a_sizes[proc_number] += 3 * self.shared_arrays["number_of_atoms"].array[i]
            if self.fitsnap_dict["stress"]:
                sub_a_sizes[proc_number] += 6
        assert sum(sub_a_sizes) == len(self.shared_arrays['a'].array)
        self.add_2_fitsnap("sub_a_size", sub_a_sizes)
        self._bcast_fitsnap("sub_a_size")
        self.fitsnap_dict["sub_a_size"] = sub_a_sizes[self._sub_rank]

        count = 0
        indices = np.zeros((self._sub_size, 2), dtype=np.int)
        for i, value in enumerate(sub_a_sizes):
            indices[i] = count, count+value-1
            count += value
        self.add_2_fitsnap("sub_a_indices", indices)
        self._bcast_fitsnap("sub_a_indices")
        self.fitsnap_dict["sub_a_indices"] = indices[self._sub_rank]

    @stub_check
    def combine_coeffs(self, coeff):
        new_coeff = None
        if self._sub_rank == 0:
            new_coeff = self._head_group_comm.allreduce(coeff) / self._number_of_nodes
        return self._sub_comm.bcast(new_coeff)

    @staticmethod
    def get_ram():
        mem = virtual_memory()
        return mem.total

    def set_logger(self, logger):
        self.logger = logger

    def pytest_is_true(self):
        self.pytest = True

    def abort(self):
        self._comm.Abort()

    def exception(self, err):
        self.killer.already_killed = True

        if pt.logger is None and self._rank == 0:
            raise err

        pt.close_lammps()
        if self._rank == 0:
            self.logger.exception(err)
            if pt.pytest:
                raise err

        sleep(5)
        if self._comm is not None:
            self.abort()

    # Where The Object Oriented Magic Happens
    # from files in this_file's directory import subclass of this_class
    @staticmethod
    def get_subclasses(this_name, this_file, this_class):
        # Reset Path cls to remove old cls paths
        from pathlib import Path

        name = this_name.split('.')
        main_dir = Path(this_file).resolve().parent
        paths = [main_dir]
        for path in main_dir.iterdir():
            if path.is_dir():
                paths.append(path)
        for package_dir in paths:
            for (_, module_name, c) in iter_modules([str(package_dir)]):
                if module_name != name[-1] and module_name != name[-2]:
                    temp_name = name[:-1]
                    the_path = str(package_dir).split("/")
                    index = the_path.index(name[-2])
                    if index != len(the_path)-1:
                        temp_name.extend(the_path[index+1:])
                    module = import_module(f"{'.'.join(temp_name)}.{module_name}")
                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)

                        if isclass(attribute) and issubclass(attribute, this_class) and attribute is not this_class:
                            # Add the class to this package's variables
                            globals()[attribute_name] = attribute


class DistributedList:
    """
    A class to wrap python list to ensure size stays the same allowing collection at end.

    ...

    Attributes
    ----------
    _len : int
        length of distributed list held by current proc

    _list : list
        local section of distributed list

    Methods
    ----------
    get_list():
        returns deepcopy of internal list

    """

    def __init__(self, proc_length):
        self._len = proc_length
        self._list = list(" ") * self._len

    def __getitem__(self, item):
        """ Return list element """
        return self._list.__getitem__(item)

    def __len__(self):
        """ Return length of list held by proc """
        return self._len

    def __setitem__(self, key, value):
        """ Set list element """
        if isinstance(key, int):
            assert len(value) == 1
            assert key <= self.__len__()
        elif isinstance(key, slice):
            # value must be list
            assert isinstance(value, list)
            # length of value must equal length of slicing
            assert len(value) == len(range(*key.indices(self.__len__())))
            # slice ending must not exceed Distributed list bound
            assert key.stop <= self.__len__()
        else:
            raise NotImplementedError("Indexing type {} for Distributed list is not impelemented".format(type(key)))
        self._list.__setitem__(key, value)

    def __repr__(self):
        """ Print list """
        return self._list.__repr__()

    def get_list(self):
        """ Return list """
        return deepcopy(self._list)


class SharedArray:

    def __init__(self, size1, size2=1, dtype='d', multinode=0, comms=None):

        # total array for all procs
        self.array = None
        # sub array for this proc
        self.sliced_array = None

        self.energies_index = None
        self.forces_index = None
        self.strain_index = None

        self._length = size1
        self._scraped_length = self._length
        self._total_length = self._length
        self._node_length = None
        self._width = size2

        # These are sub comm and sub rank
        # Comm, sub_com, head_node_comm
        # comm, rank, size
        self._comms = comms

        if multinode:
            self.multinode_lengths()

        if dtype == 'd':
            item_size = MPI.DOUBLE.Get_size()
        elif dtype == 'i':
            item_size = MPI.INT.Get_size()
        else:
            raise TypeError("dtype {} has not been implemented yet".format(dtype))
        if self._comms[1][1] == 0:
            self._nbytes = self._length * self._width * item_size
        else:
            self._nbytes = 0

        # win = MPI.Win.Allocate_shared(self._nbytes, item_size, Intracomm_comm=self._comms[1][0])
        win = MPI.Win.Allocate_shared(self._nbytes, item_size, comm=self._comms[1][0])

        buff, item_size = win.Shared_query(0)

        if dtype == 'd':
            assert item_size == MPI.DOUBLE.Get_size()
        elif dtype == 'i':
            assert item_size == MPI.INT32_T.Get_size()
        if self._width == 1:
            self.array = np.ndarray(buffer=buff, dtype=dtype, shape=(self._length, ))
        else:
            self.array = np.ndarray(buffer=buff, dtype=dtype, shape=(self._length, self._width))

    def get_memory(self):
        return self._nbytes

    def get_storage_length(self):
        # Maximum length of A storage on node
        return self._length

    def get_scraped_length(self):
        # Length of A scraped by node
        return self._scraped_length

    def get_node_length(self):
        # Length of A owned by current node
        return self._node_length

    def get_total_length(self):
        # True Length of A
        return self._total_length

    def multinode_lengths(self):
        # Each head node needs to have mb or its scraped length if longer
        # Solvers which require this: ScaLAPACK
        remainder = 0
        self._scraped_length = self._length
        if self._comms[1][1] == 0:
            self._total_length = self._comms[2][0].allreduce(self._scraped_length)
            # mb is the floored average array length, extra elements are dumped into the first array
            self._node_length = int(np.floor(self._total_length / self._comms[2][2]))
            if self._comms[2][1] == 0:
                remainder = self._total_length - self._node_length*self._comms[2][2]
            self._node_length += remainder
        self._total_length = self._comms[1][0].bcast(self._total_length)
        self._node_length = self._comms[1][0].bcast(self._node_length)
        self._length = max(self._node_length, self._scraped_length)


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

    def get_memory(self):
        return self.array.nbytes


class Output:

    def __init__(self):
        self.none = None

    @staticmethod
    def screen(*args, **kw):
        pt.single_print(*args, **kw)

    @staticmethod
    def exception(err):
        # There is almost never a reason to use this
        pt.exception(err)


if __name__.split(".")[-1] == "parallel_tools":
    pt = ParallelTools()
    if stubs == 0:
        double_size = MPI.DOUBLE.Get_size()
    else:
        double_size = ctypes.sizeof(ctypes.c_double)
    output = Output()
