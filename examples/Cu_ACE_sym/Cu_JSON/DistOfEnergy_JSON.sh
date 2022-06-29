#!/bin/bash
k=1
for i in AIMD Surface Vacancy Elastic
do 
files=`ls -ltrh $i/*.json | wc | awk '{print($1)}'`
echo "#ID Energy/atom" > EnergyHistogram_$i.dat
for j in `ls $i/*.json | sed ':a;N;$!ba;s/\n/ /g'`
do
let k=$k+1
sed s/\"/\\n/g $j | grep -w -A1 "Energy" | sed s/\://g | sed s/,//g | tail -n1 | awk -F " " '{print($1)}' > tmp
totene=`awk '{print($1)}' tmp`
sed s/\"/\\n/g $j | grep -w -A1 "NumAtoms" | sed s/\://g | sed s/,//g | tail -n1 | awk -F " " '{print($1)}' > tmp
numat=`awk '{print($1)}' tmp`
#eneperat=`echo "scale=6; $totene / $numat" |  bc`
echo $j $totene $numat >> EnergyHistogram_$i.dat
done 
done
cat EnergyHistogram_* > EnergyHistogram.dat
rm -f EnergyHistogram_*.dat tmp
