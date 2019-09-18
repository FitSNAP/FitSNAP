#!/bin/bash
fraction=$1
tar -xf WBe_FullTrainingJSON_SimpleMultiElem.tar
rm -f grouplist.tmp
cd JSON

for i in 001FreeSurf Defect_BCrowd  Defect_Vacancy  Divacancy ElasticDeform_Shear  gamma_surface  vacancy 010FreeSurf Defect_BOct DFT_MD_1000K Elast_BCC_Shear  ElasticDeform_Vol  gamma_surface_vacancy  WSurface_BeAdhesion 100FreeSurf Defect_BSplit  DFTMD_1000K Elast_BCC_Vol EOS  Liquids BCC_ForceLib_W110  Defect_BTetra  DFT_MD_300K Elast_FCC_Shear  EOS_BCC  md_bulk BCC_ForceLib_W111  Defect_Crowd DFTMD_300K  Elast_FCC_Vol EOS_Data slice_sample BCC_ForceLib_WOct  Defect_Oct dislocation_quadrupole  Elast_HCP_Shear  EOS_FCC  StackFaults BCC_ForceLib_WTet  Defect_Tet Disordered_Struc Elast_HCP_Vol EOS_HCP  surface
do
k=0
rm -rf in_$i ; mkdir in_$i
rm -rf out_$i ; mkdir out_$i
countin=0
countout=0
files=`ls -ltrh $i/*.json | wc | awk '{print($1)}'`
limit=` echo " ($files * $fraction) / 1 "| bc `
outlimit=` echo " ($files * (1.0 - $fraction)) / 1 "| bc `
for j in `ls $i/*.json | sed ':a;N;$!ba;s/\n/ /g'`
do
let k=$k+1
dice=`echo "scale=6 ; $RANDOM/32767" | bc`
verdict=`echo " ($dice < $fraction) / 1" | bc`
#echo $i $files $limit $outlimit $dice $verdict
if [ $verdict -eq 1 ] && [ $countin -lt $limit ]; then
mv $j in_$i
countin=`ls -ltrh in_$i/*.json | wc -l `
elif [ $countout -ge $outlimit ]; then
mv $j in_$i
countin=`ls -ltrh in_$i/*.json | wc -l `
else
mv $j out_$i
countout=`ls -ltrh out_$i/*.json | wc -l `
fi
done
echo in_$i  $countin  >> grouplist.tmp
echo out_$i $countout >> grouplist.tmp
done
mv grouplist.tmp ../
rmdir 001FreeSurf Defect_BCrowd  Defect_Vacancy  Divacancy ElasticDeform_Shear  gamma_surface  vacancy 010FreeSurf Defect_BOct DFT_MD_1000K Elast_BCC_Shear  ElasticDeform_Vol  gamma_surface_vacancy  WSurface_BeAdhesion 100FreeSurf Defect_BSplit  DFTMD_1000K Elast_BCC_Vol EOS  Liquids BCC_ForceLib_W110  Defect_BTetra  DFT_MD_300K Elast_FCC_Shear  EOS_BCC  md_bulk BCC_ForceLib_W111  Defect_Crowd DFTMD_300K  Elast_FCC_Vol EOS_Data slice_sample BCC_ForceLib_WOct  Defect_Oct dislocation_quadrupole  Elast_HCP_Shear  EOS_FCC  StackFaults BCC_ForceLib_WTet  Defect_Tet Disordered_Struc Elast_HCP_Vol EOS_HCP  surface
cd ../
