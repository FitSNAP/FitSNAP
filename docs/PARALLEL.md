# Parallel Features

FitSnap3 consists of three main phases:

1) Scraping the data from JSON Files and reading configuration files.
2) Using LAMMPS to compute bispectrum coefficients for SNAP.
3) Solving for SNAP coefficients.

Currently, only step 2 is set up for parallel calculation.

There are two mechanisms for parallelization: Multiprocessing and MPI.
The former is simpler, but limited to one node.
However, if only using one node, Multiprocessing will be slightly faster than MPI.

Scaling step 2 over more and more cores will eventually (10-20 cores) cause the total running time to be dominated by the serial steps 1 and 3 (Amdahl's Law).
Running step 2 in parallel is most helpful for larger datasets with large numbers of bispectrum components (large `twojmax`).
There are also communication overheads associated with parallelization, as the simple map strategy used
involves all-to-one type communication patterns.

## Multiprocessing Parallelization

- `Multiprocessing` comes as part of the python standard library; no installation needed.
- Use the `-j <N>` or `--jobs <N>` argument to set the number of processes.
Python's Multiprocessing module will be used to start that many process, and in each one an instance of LAMMPS.
- This form of Parallelization can thus only be used on a single node.
- It will most likely not help to set the number of jobs any larger than the number of cores available.

## MPI Parallelization

- Install `mpi4py`, version 3.0.0 or greater. (Tested with 3.0.1)
- Tested with `mpich/3.1.3 gcc_4.8.5` and `intel-mpi intel` mpi & compiler setups.
- Launch FitSnap3 with mpiexec using **1** initial process.
- Send FitSnap3 the flag `--mpi` **in addition to** `-j <N>`, where `<N>` is the total number of processes.
- FitSnap3 will use mpi4py's MPIPoolExecutor class to spawn N new processes across the allocated nodes,
and farm out the LAMMPS calculation to these processes. 
- Example call: Assuming the job has been allocated 32 nodes with 16 cores per node

      `mpiexec -n 1 python -m fitsnap3 examples/WBe_PRB2019/WBe-example.in -j 512 --mpi`
- Note that if you are working on one node, MPI and Multiprocessing should give identical performance.

## Other aspects of Parallelization:

### Chunk sizes

`fitnsap3/deploy.py` contains a global variable `DEFAULT_CHUNKSIZE` that controls the number
of configurations which each process will consume before communicating results back to the main process.
If you find yourself wanting to tune your parallelization for best performance, this variable can play a role.
At lower chunk sizes, communication will be more frequent and expensive,
but possibly allowing for better load-balancing between processes.
At higher chunk sizes, communication will be less frequent and more efficient,
but load distribution between processes might suffer.

For typical fits running on typical hardware, the default value of DEFAULT_CHUNKSIZE=10
yields close to 100% efficiency, and there is no need to change it. 

### LAMMPS Parallelization

FitSnap3's parallelization mechanisms work orthogonally to any parallelization in LAMMPS,
e.g. OpenMP thread parallelization. If you do not use any parallelization in LAMMPS,
you should probably set the number of FitSNAP processes `<N>` to the number of cores/ranks available.
 

