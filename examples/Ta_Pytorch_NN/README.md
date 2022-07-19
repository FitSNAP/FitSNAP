### PyTorch neural network force fitting example.

Run the bash script with `./run.sh` to run FitSNAP and fit to forces.

This bash script also creates corresponding plots of errors vs. epochs and target force vs. model force. 

To check that the model forces match those calculated with finite difference, do `python fd_force_check.py`, which creates a `fd_force_check.png` plot. This finite difference script uses the same force calculation routine that we use in `fitsnap3lib/lib/neural_networks/pytorch.py`

The `performance.dat` file contains data points associated with performance; first column is total number of atoms in the training set and second column is time(s)/epoch. This was done without pruning the neighbor list.

When fitting forces AND energies, it was found to best have `energy_weight=1e-4` and `force_weight=1.0` or some similar ratio. 
