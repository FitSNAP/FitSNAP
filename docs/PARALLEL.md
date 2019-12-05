# Parallel Features

FitSnap3 consists of three main phases:

1) Scraping the data from JSON Files and reading configuration files.
2) Using LAMMPS to compute bispectrum coefficients for SNAP.
3) Solving for SNAP coefficients.

Currently, only step 2 is set up for parallel calculation.

There are two mechanisms for parallelization: Multiprocessing and MPI.
The former is simpler, but limited to one node.
However, if only using one node, Multiprocessing will be slightly faster than MPI.

Because steps 1 and 3 are still in serial, there are limits on how useful parallelization can be.
It will be most helpful for larger datasets with large numbers of bispectrum components (large `twojmax`).
There are also communication overheads associated with parallelization, as the simple map strategy used
involves all-to-one type communication patterns.

## Multiprocessing Parallelization

- `Multiprocessing` comes as part of the python standard library; no installation needed.
- Use the `-j <N>` or `--jobs <N>` argument to set a number of processes.
Python's Multiprocessing module will be used to start that many process, and in each one an instance of LAMMPS.
- This form of Parallelization can thus only be used on a single node.
- It will most likely not help to set the number of jobs any larger than the number of cores available.

## MPI Parallelization

- Install `mpi4py`, version 3.0.0 or greater. (Tested with 3.0.1)
- Tested with `mpich/3.1.3 gcc_4.8.5` and `intel-mpi intel` mpi & compilier setups.
- Launch FitSnap3 with mpiexec using **1** initial process.
- Send FitSnap3 the flag `--mpi` **in addition to** `-j <N>` for N ranks.
- FitSnap3 will use mpi4py's MPIPoolExecutor class to spawn N new processes across the allocation,
and farm out the LAMMPS calculation to these processes. 
- Example call: `mpiexec -n 1 python -m fitsnap3 examples/WBe_PRB2019/WBe-example.in -j 512 --mpi`
- Note that if you are working on one node, there is no known advantage to using MPI over multiprocessing.


## Other aspects of Parallelization:

### Chunk sizes

`fitnsap3/deploy.py` constains a global variable `DEFAULT_CHUNKSIZE` that controls the number
of configurations which each process will consume before communicating results back to the main process.
If you find yourself wanting to tune your parallelization for best performance, this variable can play a role.
At lower chunk sizes, communication will be more frequent and expensive,
but possibly allowing for better load-balancing between processes.
At higher chunk sizes, communication will be less frequent and more efficient,
but load distribution between processes might suffer.

We don't expect the variable to play a huge role in practical situations, however,
if you do find that tuning the chunksize plays a significant role in the fit 

 

### LAMMPS Parallelization

FitSnap3's parallelization mechanisms work orthogonally to any parallelization in LAMMPS,
e.g. OMP thread parallelization.If you do not use any parallelization in LAMMPS,
you should probably set the number of processes to parallelize to the number of cores/ranks available.
 

