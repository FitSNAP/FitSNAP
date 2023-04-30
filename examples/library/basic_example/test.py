from mpi4py import MPI 
import numpy as np 

comm = MPI.COMM_WORLD 

# create a shared array of size 1000 elements of type double
size = 1000 
itemsize = MPI.DOUBLE.Get_size() 
if comm.Get_rank() == 0: 
    nbytes = size * itemsize 
else: 
    nbytes = 0

# on rank 0, create the shared block
# on rank 1 get a handle to it (known as a window in MPI speak)
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 

# create a numpy array whose data points to the shared mem
buf, itemsize = win.Shared_query(0) 
assert itemsize == MPI.DOUBLE.Get_size() 
ary = np.ndarray(buffer=buf, dtype='d', shape=(size,)) 

# in process rank 1:
# write the numbers 0.0,1.0,..,4.0 to the first 5 elements of the array
if comm.rank == 1: 
    ary[:5] = np.arange(5)

# wait in process rank 0 until process 1 has written to the array
comm.Barrier() 

# check that the array is actually shared and process 0 can see
# the changes made in the array by process 1
if comm.rank == 0: 
    print(ary[:10])