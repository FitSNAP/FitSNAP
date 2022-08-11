### Example on how to run an MD simulation with the PyTorch neural network potential in LAMMPS.

1) If the ML-IAP package was loaded successfully during fitting, you should have a ".pt" model that is saved at the end of the simulation.
Note: This is not the "Ta_Pytorch.pt" model that is saved after every epoch.

2) This is a lammps-ready model you can directly use with LAMMPS.
Run `python mliap_pytorch_Ta.py` to start an example simulation with the provided potential. The ".descriptor" contains the SNAP definition and must be in the same directory.
