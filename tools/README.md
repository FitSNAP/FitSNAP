#Misc analysis tools

DistOfEnergy_JSON.sh - Simple script that pulls out the energies from JSON training data. Useful when characterizing what is in the training sets, and looking for outlier configurations.


ShiftEnergy.sh - A bash script (see a pattern here?) to adjust the energy per atom in your training data. This also can be done using the Eshift1..Eshift4 input variables in FitSNAP, but this will be applied aross all of your training of that particular element type. This script is nice if you are trying to merge training sets from DFT that use different pseudopotentials, and thus have a different reference energy per atom. 


default_ACE_settings.py - Provides ball-park starting points for ACE hyperparameters based on chemical information. For example, radial cutoffs are chosen based on a number of van der waals shells away from the central atom. The suggested 'lambda' hyperparameters are determined by more pragmatic means; they are chosen based on a specific fraction of the radial cutoff that does not cause radial functions to decay too quickly. Hard core repulsions (below rcinner) are set based on a small fraction (0.25) of the smallest ionic bond distance. It also provides suggested ZBL hyperparameters per bond type, assuming ZBL repulsions take over from 0.25r<sub>ionic</sub><sup>min.</sup> to r<sub>ionic</sub><sup>min.</sup>. These suggested hyperparameters are by no means perfect and should <i>not</i> be taken ase the 'best choice' for every system. This script <i>DOES</i> serve as a ball-park estimator for ranges of hyperparameters using physical motivation. It is ultimately still up to the user to optimize hyperparameters and select reasonable ranges per system and training set.
