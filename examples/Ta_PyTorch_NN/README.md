### PyTorch neural network force fitting example.

Fit a NN with

    python -m fitsnap3 Ta-example.in --overwrite # use --overwrite if you want to overwrite existing
fitsnap files.

This also creates corresponding plots of errors vs. epochs and target force vs. model force. 

To check that the model forces match those calculated with finite difference, do 
`python fd_force_check.py`, which creates a `fd_force_check.png` plot. 
This finite difference script uses the same force calculation routine that we use in 
`fitsnap3lib/lib/neural_networks/pytorch.py`

When fitting forces AND energies, it was found to best have `energy_weight=1e-2` and 
`force_weight=1.0` or some similar ratio.

### Calculating fitting errors.

FitSNAP produces output files that we can use to calculate error ourselves. To do this, run

    python plot_comparison.py

which will calculate errors and plot comparisons with the detailed fitting data. 

### Running MD with NN potential.

Refer to folder "MD" for instructions to run MD simulations with the potential.

### Evaluating energies/forces in Python.

Sometimes you want to calculate energies/forces on another set using an already fitted model. In this 
case it is wasteful to re-calculate the descriptors. We can therefore load a pickled list of 
Configuration objects which are used by FitSNAP for NN fitting. First generate the pickled list 
of Configuration objects `configs.pickle` by performing a fit, then use the script 
`evaluate_configs.py` which will load this pickled list and calculate energies/forces for all 
configs.
