#!/bin/sh 
module purge
module load dakota/dakota-6.5
module swap mkl/17.0
num=$(echo $1 | awk -F. '{print $NF}')
CONCURRENCY=20
PPN=36
APPLIC_PROCS=36
applic_nodes=$(( ($APPLIC_PROCS+$PPN-1) / $PPN ))
relative_node=$(( (num - 1) % CONCURRENCY * APPLIC_PROCS / PPN ))
node_list="+n${relative_node}"
for node_increment in `seq 1 $((applic_nodes - 1))`; do
  node_list="$node_list,+n$((relative_node + node_increment))"
done
echo FAIL > $2

dprepro $1 Ta_Trial.template Ta_Trial.in 
dprepro $1 grouplist.template grouplist.in
sed -e "s/NODE_LIST/$node_list/" Ta_Trial.in > input.in
time ../../src/fitsnap.py < input.in  > fitsnap.screen.out

rm -rf DumpSnap/
rm -rf snap/
rm -f  abtotal.dat predictions.json SNAPforce.dat *.table SNAPvirial.dat

tail -1 SNAPenergy_MeanError.dat | awk '{print $NF}' > tmp.out
tail -1 SNAPforce_MeanError.dat | awk '{print $NF}' >> tmp.out
#tail -1 SNAPvirial_MeanError.dat | awk '{print $NF}' >> tmp.out

Eerr=`tail -1 SNAPenergy_MeanError.dat | awk '{print $NF}'`
Ferr=`tail -1 SNAPforce_MeanError.dat | awk '{print $NF}'`
#Verr=`tail -1 SNAPvirial_MeanError.dat | awk '{print $NF}'`

exe="/ascldap/users/mitwood/LAMMPS_Builds/svn_lammps/trunk/src/lmp_Dakota_KKmpi_5.23"

###########Elastic Constants#######################
mpiexec -np $APPLIC_PROCS -host $node_list ${exe}  -in in.elastic -screen none >& ./run_script.out

grep "^Lattice Constant abcc" log.lammps >> ./run_script.out
grep "^Elastic Constant C11bcc" log.lammps >> ./run_script.out
grep "^Elastic Constant C12bcc" log.lammps >> ./run_script.out
grep "^Elastic Constant C44bcc" log.lammps >> ./run_script.out
latcerr=`grep "^Lattice Constant abcc" log.lammps  | awk '{print(100*(((1-$5/3.316)**(2.0))**(1.0/2.0)))}'`
c11=`grep "^Elastic Constant C11bcc" log.lammps | awk '{print($5)}'`
c12=`grep "^Elastic Constant C12bcc" log.lammps | awk '{print($5)}'`
c1112err=`awk '{print(100*(((1-((1.0*'"$c11"')-(1.0*'"$c12"'))/101.4)**(2.0))**(1.0/2.0)))}' tmp.out | tail -n1`
c44err=`grep "^Elastic Constant C44bcc" log.lammps | awk '{print(100*(((1-$5/75.3)**(2.0))**(1.0/2.0)))}'`
str_err=`awk '{print('"$latcerr"'+'"$c1112err"'+'"$c44err"')}' tmp.out | tail -n1`

############Defect Formation Energies######################
#echo "# Type UnRelaxTot RelaxTot RelaxPerAt" > tmp_defectE.dat
j=0
rm -f defectE.dat tmp_defectE.dat 
for i in perf oct 110d vac
do
let j=$j+1
mpiexec -np $APPLIC_PROCS -host $node_list  ${exe} -in in.snap_run0_${i} > log.${i}
rperf_petot=`grep "RELAXED PE PEATOM" log.perf | awk '{print($4)}'`
rperf_peatom=`grep "RELAXED PE PEATOM" log.perf | awk '{print($5)}'`
rel_petot=`grep "RELAXED PE PEATOM" log.lammps | awk '{print($4)}'`
rel_peatom=`grep "RELAXED PE PEATOM" log.lammps | awk '{print($5)}'`

if [ ${#rel_petot} -gt 0 ]; then
echo $j $i $rel_petot $rperf_petot $rperf_peatom >> tmp_defectE.dat
else
echo $j $i " PEATOM 0.0 " $rperf_petot $rperf_peatom >> tmp_defectE.dat
fi
done
mv tmp_defectE.dat defectE.dat 

input=defectE.dat

#W defects
ecoh=`awk '{if($1==1)print(($6))}' $input`
perecoh=`awk '{print(100*((((1+('"$ecoh"'/8.10))**(2.0))**(1.0/2.0))))}' $input | tail -n1`
woct=`awk '{if($1==2)print(($4-$5-(1.0*$6)))}' $input`
perwoct=`awk '{print(100*((((1-('"$woct"'/3.01))**(2.0))**(1.0/2.0))))}' $input | tail -n1`
w110d=`awk '{if($1==3)print(($4-$5-(1.0*$6)))}' $input`
perw110=`awk '{print(100*((((1-('"$w110d"'/5.63))**(2.0))**(1.0/2.0))))}' $input | tail -n1`
wvac=`awk '{if($1==4)print(($4-$5+(1.0*$6)))}' $input`
perwvac=`awk '{print(100*((((1-('"$wvac"'/2.89))**(2.0))**(1.0/2.0))))}' $input | tail -n1`

totWDefErr=`awk '{print('"$perwoct"'+'"$perw110"'+'"$perwvac"')}' $input | tail -n1`

if [ ${#str_err} -gt 0 ]; then
echo $str_err >> tmp.out
else
echo "1000.0" >> tmp.out
fi
#if [ ${#totWDefErr} -gt 0 ]; then
#echo $totWDefErr >> tmp.out
#else
#echo "1000.0" >> tmp.out
#fi
if [ ${#perecoh} -gt 0 ]; then
echo $perecoh >> tmp.out
else
echo "1000.0" >> tmp.out
fi
if [ ${#perwoct} -gt 0 ]; then
echo $perwoct >> tmp.out
else
echo "1000.0" >> tmp.out
fi
if [ ${#perwvac} -gt 0 ]; then
echo $perwvac >> tmp.out
else
echo "1000.0" >> tmp.out
fi
if [ ${#perw110} -gt 0 ]; then
echo $perw110 >> tmp.out
else
echo "1000.0" >> tmp.out
fi
echo $Eerr $Ferr $str_err $totWDefErr $rperf_peatom >> trimmed_opt.dat
mv tmp.out $2

