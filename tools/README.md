#Misc analysis tools

DistOfEnergy_JSON.sh - Simple script that pulls out the energies from JSON training data. Useful when characterizing what is in the training sets, and looking for outlier configurations.


ShiftEnergy.sh - A bash script (see a pattern here?) to adjust the energy per atom in your training data. This also can be done using the Eshift1..Eshift4 input variables in FitSNAP, but this will be applied aross all of your training of that particular element type. This script is nice if you are trying to merge training sets from DFT that use different pseudopotentials, and thus have a different reference energy per atom. 
