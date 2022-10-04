### Example on how to run an MD simulation with the PyTorch neural network potential in LAMMPS.

First fit a model in the prevous directory, which should save a `FitTorch_pytorch.pt` file.

**Before running LAMMPS,** be sure to set your PYTHONPATH to include your `site-packages` 
directory. If you don't know where your `site-packages` directory is, run the following lines in
your python interpreter:

    import site
    print(site.getsitepackages())

This will help the embedded Python in ML-IAP find your torch and other libraries.

Run the LAMMPS input script with

    mpirun -np 2 lmp < in.run

The `.descriptor` file contains the SNAP definition and must be in the same directory.

Alternatively, we run LAMMPS MD in Python. **This only works if your Python interpreter is 
compatible with ML-IAP, which is unlikely. We recommend sticking with the the usual use of LAMMPS
input scripts for now.** Run MD in python with

    mpirun -np 2 python mliap_pytorch_Ta.py
