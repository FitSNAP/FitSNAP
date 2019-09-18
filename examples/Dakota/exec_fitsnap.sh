#!/bin/sh 
echo FAIL > $2
pwd
dprepro $1 Ta_Trial.template Ta_Trial.in 
dprepro $1 grouplist.template grouplist.in
time ../../src/fitsnap.py < Ta_Trial.in  > fitsnap.screen.out

rm -rf DumpSnap/
rm -rf snap/
rm -f  abtotal.dat predictions.json SNAPforce.dat *.table SNAPvirial.dat

tail -1 SNAPenergy_MeanError.dat | awk '{print $NF}' > tmp.out
tail -1 SNAPforce_MeanError.dat | awk '{print $NF}' >> tmp.out
#tail -1 SNAPvirial_MeanError.dat | awk '{print $NF}' >> tmp.out

Eerr=`tail -1 SNAPenergy_MeanError.dat | awk '{print $NF}'`
Ferr=`tail -1 SNAPforce_MeanError.dat | awk '{print $NF}'`
#Verr=`tail -1 SNAPvirial_MeanError.dat | awk '{print $NF}'`

exe="/ascldap/users/mitwood/Documents/LAMMPS_Builds/trunk/src/lmp_kokkos_mpi_only"

###########Elastic Constants#######################
time mpiexec -np 16 ${exe}  -in in.elastic -screen none >& ./run_script.out

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
j=0
rm -f defectE.dat tmp_defectE.dat 
for i in perf 
do
let j=$j+1
time mpiexec -np 36 ${exe} -in in.snap_run0_${i} > log.${i}
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

ecoh=`awk '{if($1==1)print(($6))}' $input`
perecoh=`awk '{print(100*((((1+('"$ecoh"'/8.10))**(2.0))**(1.0/2.0))))}' $input | tail -n1`

if [ ${#str_err} -gt 0 ]; then
echo $str_err >> tmp.out
else
echo "1000.0" >> tmp.out
fi
if [ ${#perecoh} -gt 0 ]; then
echo $perecoh >> tmp.out
else
echo "1000.0" >> tmp.out
fi
mv tmp.out $2

rm -f lastcol pot_def_Ta.mod param.mod init.mod tmp.lammps.variable Ta_Trial.template in.snap_run0_perf in.elastic grouplist.template grouplist.out fitsnap.py displace.mod run_fitsnap potential.mod params.in.1 in.snap Ta_Trial.in lmpoutput.inc SNAPenergy_MeanError.dat SNAPforce_MeanError.dat SNAPvirial_MeanError.dat pot_Ta.mod SNAPcoeff.dat fitsnap.screen.out restart.equil run_script.out log.perf log.lammps defectE.dat                                        

