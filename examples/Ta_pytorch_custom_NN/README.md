### PyTorch neural network force fitting example.

Fit a NN with

    python -m fitsnap3 Ta-example.in --overwrite # use --overwrite if you want to overwrite existing
fitsnap files.

To check that the model forces match those calculated with finite difference, do 
`python fd_force_check.py`, which creates a `fd_force_check.png` plot. 
This finite difference script uses the same force calculation routine that we use in 
`fitsnap3lib/lib/neural_networks/pytorch.py`

When fitting forces AND energies, it was found to best have `energy_weight=1e-2` and 
`force_weight=1.0` or some similar ratio.

### Calculating fitting errors.

FitSNAP produces output files that we can use to calculate error ourselves. To do this, run

    python calculate_fitting_errors.py

which will calculate mean absolute errors for energy and forces, as well as plot a distribution of
force errors, for the training and validation sets. 

### Running MD with custom pairwise NN potential in MLIAP Unified.

Deploy the model to MLIAP Unified with:

    python deploy-script.py

Then run MD with:

    mpirun -np P lmp -in in.run


