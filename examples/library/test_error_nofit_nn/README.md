### Evaluate fit with NN potential in LAMMPS

First generate LAMMPS data files with

    python xyz2data.py

Then evaluate forces with those data files with

    python evaluate.py

TODO: Convert this to a FitSNAP library use, similar to the `test_error_nofit` example for linear. 
