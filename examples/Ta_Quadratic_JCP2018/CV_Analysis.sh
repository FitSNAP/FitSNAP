#!/bin/bash
for k in Displaced_A15 Displaced_BCC Displaced_FCC Elastic_BCC Elastic_FCC GSF_110 GSF_112 Liquid Surface Volume_A15 Volume_BCC Volume_FCC
do
rm -f ${k}_*.dat
  for j in $(seq 0.05 0.05 0.95)
    do
      for i in Energy Force Stress
      do
      grep ${k} TrainFrac_${j}/Ta_metrics_Run* | grep 'CVTrain_Unweight' | grep ${i} | awk -F "," '{print($1,$2,$3,$4,$5)}' > TrainErr.dat
      grep ${k} TrainFrac_${j}/Ta_metrics_Run* | grep 'CVTest_Unweight' | grep ${i} | awk -F "," '{print($1,$2,$3,$4,$5)}' > TestErr.dat
      paste TrainErr.dat TestErr.dat | awk '{print($5,$10,$4,$9)}' > ${k}_${j}_${i}.dat
      avgtest=`awk '{ if(NF>0) (sum+=$2 n++)} END { print(n,sum,sum/n); }' ${k}_${j}_${i}.dat | awk '{print($3)}'`
      avgtrain=`awk '{ if(NF>0) (sum+=$1 n++)} END { print(n,sum,sum/n); }' ${k}_${j}_${i}.dat| awk '{print($3)}'`
      reps=`wc -l ${k}_${j}_${i}.dat |  awk '{print($1)}'`
      ntrain=`awk '{ if(NF>0) (sum+=$3 n++)} END { print(n,sum,sum/n); }' ${k}_${j}_${i}.dat | awk '{print($3)}'`
      ntest=`awk '{ if(NF>0) (sum+=$4 n++)} END { print(n,sum,sum/n); }' ${k}_${j}_${i}.dat | awk '{print($3)}'`
      stdtest=`awk '{ if(NF>0) (sum+=(((($1-'"$avgtest"')**2)/('"$reps"'-1))**0.50) n++)} END { print(sum); }' ${k}_${j}_${i}.dat`
      stdtrain=`awk '{ if(NF>0) (sum+=(((($1-'"$avgtrain"')**2)/('"$reps"'-1))**0.50) n++)} END { print(sum); }' ${k}_${j}_${i}.dat`
      echo $j $ntrain $ntest $avgtrain $avgtest $stdtrain $stdtest >> ${k}_${i}.dat
      rm -f TestErr.dat TrainErr.dat ${k}_${j}_${i}.dat
    done
  done
done
