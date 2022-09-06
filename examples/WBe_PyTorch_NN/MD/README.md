### LAMMPS NN-SNAP MD

First fit a model in the prevous directory, which should save a `FitTorch_pytorch.pt` file.

Now we are ready to run MD with this model here, with

    mpirun -np 4 python mliap_pytorch_WBe.py
