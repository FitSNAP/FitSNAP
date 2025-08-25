# Parallel Features

FitSnap3 consists of three main phases:

1) Scraping from data files, such as JSON, and reading configuration files.
2) Using LAMMPS to compute bispectrum coefficients for SNAP.
3) Solving for SNAP coefficients.

All steps are set up for parallel calculation.

FitSnap3 is a hybrid MPI and threading program.
The scraping of data files and calculating of fitting data is split amongst all MPI processors.
The fitting data is written into shared memory. Head nodes of each shared memory pool then solve
their respective linear equations with threading or GPUs depending on the choice of solver.

## MPI Parallelization

- Install `mpi4py`, version 3.0.0 or greater. (Tested with 3.0.3)
- Tested with `openmpi-2.0.4 and above` mpi & compiler setups.
- Launch FitSnap3 with mpiexec using **1** initial process.
- Example call: Assuming the job has been allocated 2 nodes with 16 cores per node

      `mpiexec -N 2 -n 32 python -m fitsnap3 examples/WBe_PRB2019/WBe-example.in`
