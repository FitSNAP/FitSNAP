# Memory efficient linear fits

Here we use a transpose trick to significantly reduce the size of the A matrix. Run a fit with:

    mpirun -np P python example.py

And use the LAMMPS input script `in.run` to run MD after.

See the `example.py` script for variables to change.