### PyTorch neural network force fitting example.

Run the bash script with `./run.sh` to run FitSNAP and fit to forces.

To make plots of fitting results, do:

    python plot_error_vs_epochs.py
    python plot_force_comparison.py

which creates corresponding pngs of the fitting results. 

To check that the model forces match those calculated with finite difference, do `python fd_force_check.py`, which creates a `fd_force_check.png` plot.  

See the `nn_force_check.ipynb` for a notebook that checks neural network forces calculated in PyTorch against finite difference forces, to ensure that we are calculating forces properly. 
