from mpi4py import MPI
from subprocess import run, PIPE


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

result = run(['ldd', '{}'.format(MPI.__file__)], stdout=PIPE)
print(result)
result = result.stdout.decode('utf-8').replace('\t', '').split('\n')[1:]
print(result)
print([string for string in result if 'libmpi' in string][0].split()[2])
