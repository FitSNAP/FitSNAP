### Example on how to run an MD simulation with the PyTorch neural network potential in LAMMPS.

If the ML-IAP package was loaded successfully during fitting, you should have a ".pt" model that is 
saved at the end of the simulation.

Note: This is not the `Ta_Pytorch.pt` model that is saved after every epoch. Instead it is the 
`FitTorch_pytorch.pt` file.

This is a lammps-ready model you can directly use with LAMMPS.
Run `mpirun -np 2 python mliap_pytorch_Ta.py` to start an example simulation with the provided 
potential. The `.descriptor` file contains the SNAP definition and must be in the same directory.

Currently normal LAMMPS scripts such as `in.run` do not work in some python environments, because
the ML-IAP package fails to load torch. This python example will therefore suffice for now.
