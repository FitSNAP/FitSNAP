rm Truth*
rm Weight*
rm Descrip*
rm Ta_*
rm Fit*
python -m fitsnap3 Ta-example.in
#mpirun -np 2 python -m fitsnap3 Ta-example.in
python plot_force_comparison.py
python plot_error_vs_epochs.py
