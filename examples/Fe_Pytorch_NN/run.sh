rm Truth*
rm Weight*
rm Descrip*
rm Ta_*
rm Fit*
python -m fitsnap3 Fe-example.in
python plot_force_comparison.py
python plot_energy_comparison.py
python plot_error_vs_epochs.py
