from mpi4py import MPI
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(MPI.__file__i, "on rank", rank)
print(sys.executable, "on rank", rank)
