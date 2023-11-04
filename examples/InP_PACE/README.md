## FitSnap3 Indium Phosphide example with ACE

## Important Notes
Multi-element ACE fit with reference energies (bzeroflag=1).
The 'erefs' variable in the ACE section may be used to set 
the intercept manually in the .yace potential file, but by
default, the intercept for bzeroflag=1 should be 0.0 for all
elements.

### Energy and forces only
Input file and expected output in 30Mar23_Standard.
Example of multielement ACE fit from FitSNAP. This example
requires that sklearn is installed for the ARD solver. The
priors are chosen based on variance in the (weighted)
training energies and forces. The default example in the
top directory is this fit.

### With stresses
Input file and expected output in 30Mar23_Standard_Stress.
Example of multielement ACE fit from FitSNAP. This example
requires that sklearn is installed for the ARD solver. The
priors are chosen based on variance in the (weighted)
training energies, forces, and stresses. 
