

for e in 1 2 3 4 5
do
    python -m fitsnap3 Ta-example.in --overwrite
    cp loss_vs_epochs.dat loss_vs_epochs_$e.dat
    cp force_comparison.dat force_comparison_$e.dat
    cp force_comparison_val.dat force_comparison_val_$e.dat
    cp energy_comparison.dat energy_comparison_$e.dat
    cp energy_comparison_val.dat energy_comparison_val_$e.dat
done
