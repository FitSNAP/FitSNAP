#!/bin/bash
for j in $(seq 0.05 0.05 0.95)
  do
  sed -e s/FRAC/${j}/g grouplist.template > grouplist-Ta.in
  mkdir TrainFrac_${j}
  for i in $(seq 1 1 10)
    do
    rm -f log.lammps run_script.out
    python3 -m snappy --overwrite -j32 -v Ta-example.in > screen_fitsnappy
    mv Ta_metrics.csv Ta_metrics_Run${i}.csv
    mv Ta_metrics_Run${i}.csv TrainFrac_${j}/
  done
done
