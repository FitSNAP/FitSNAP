### Example on how to run an MD simulation with the PyTorch neural network potential in LAMMPS.

If the ML-IAP package was loaded successfully during fitting, you should have a ".pt" model that is 
saved at the end of the simulation.

Note: This is not the `Ta_Pytorch.pt` model that is saved after every epoch. Instead it is the 
`FitTorch_pytorch.pt` file.

This is a lammps-ready model you can directly use with LAMMPS.

**Before running LAMMPS,** be sure to set your PYTHONPATH to include your `site-packages` 
directory. If you don't know where your `site-packages` directory is, run the following lines in
your python interpreter:

    import site
    print(site.getsitepackages())

Run the LAMMPS input script with

    lmp < in.run

Or run in python with

    mpirun -np 2 python mliap_pytorch_Ta.py

The `.descriptor` file contains the SNAP definition and must be in the same directory.
