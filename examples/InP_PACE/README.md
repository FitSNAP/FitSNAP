## FitSnap3 Indium Phosphide example with ACE

## Important Notes
As with other ACE fits, regularization is highly 
recommended. For this reason, and for adaptability to
training data, the Bayesian ARD method is recommended
specifically. ARD is used in all of these examples.
To see more examples of ARD, see the Ta example as well as
the sklearn website: <a>https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html</a>
These InP potentials are minimally tested and are here for 
demonstration purposes. Stress training with ACE models is
new and may require updates.

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
