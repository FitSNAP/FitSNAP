### PyTorch neural network force fitting example.

Run the bash script with `./run.sh` to run FitSNAP and fit to forces.

This bash script also creates corresponding plots of errors vs. epochs and target force vs. model force. 

To check that the model forces match those calculated with finite difference, do `python fd_force_check.py`, which creates a `fd_force_check.png` plot. This finite difference script uses the same force calculation routine that we use in `fitsnap3lib/lib/neural_networks/pytorch.py`
