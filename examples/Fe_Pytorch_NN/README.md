### PyTorch neural network force fitting example.

Fit a NN with

    python -m fitsnap3 Fe-example.in --overwrite # use --overwrite to overwrite existing fitsnap files.

This creates corresponding plots of errors vs. epochs and target force vs. model force. 

To check that the model forces match those calculated with finite difference, do 
`python fd_force_check.py`, which creates a `fd_force_check.png` plot. This finite difference 
script uses the same force calculation routine that we use in 
`fitsnap3lib/lib/neural_networks/pytorch.py`
