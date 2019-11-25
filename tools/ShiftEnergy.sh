#!/bin/bash
k=1
eshift=1.23456 
#This will shift the energy scale of the groups defined on $i 
for i in AllTypeTwo AllTypeThree AllTypeFour
do 
files=`ls -ltrh JSON/$i/*.json | wc | awk '{print($1)}'`
echo "#ID Energy/atom" > OldNew_EnergyScale_$i.dat
for j in `ls JSON/$i/*.json | sed ':a;N;$!ba;s/\n/ /g'`
do
let k=$k+1
sed s/\"/\\n/g $j | grep -w -A1 "Energy" | sed s/\://g | sed s/,//g | tail -n1 | awk -F " " '{print($1)}' > tmp
totene=`awk '{print($1)}' tmp`
sed s/\"/\\n/g $j | grep -w -A1 "NumAtoms" | sed s/\://g | sed s/,//g | tail -n1 | awk -F " " '{print($1)}' > tmp
numat=`awk '{print($1)}' tmp`
eneperat=`awk '{print('"$totene"'+'"$eshift"'*'"$numat"')}' tmp | tail -n1`
#`echo "scale=6; $totene + $eshift*$numat" |  bc`
echo $j $totene $numat $eneperat >> OldNew_EnergyScale_$i.dat
find $j -type f -exec sed -i 's/\"Energy\"\: '"$totene"'/"Energy": '"$eneperat"'/g' {} \;
done 
done
