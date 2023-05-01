## FitSnap3 Ta example with ACE


## Important Note
As with other ACE fits, regularization is highly
recommended. These Ta potentials are minimally tested and 
are here for demonstration purposes. This example uses
the RIDGE regression technique to apply a penalty to ACE
coefficients that are too large via the L2 norm penalty. 


### RIDGE regression
This is the default example provided for the linear ACE fit
of Ta. The ridge regressor may be used from sklearn, or 
through internal solvers. No additional installation of
sklearn is required for this example. The default input in
this top directory, and in 30Mar23_RIDGE, uses a ridge
regressor from FitSNAP by default. Otherwise, this is a 
copy of the ARD example but with a different regressor.

RIDGE regression is one of the more simple ways to obtain
a regularized solution to the least squares problem. An L2
penalty ( <b>w</b> <sup>T</sup> (α <b>I</b>) <b>w</b> ) is added
to the least squares cost function to penalize model 
weights that get too large. While this can reduce 
overfitting and stabilize models, it does little for 
sparsification/feature selection.

### RIDGE hyperparameters

alpha : (float) regularization hyperparameter that scales
the penalty for the linear model coefficients. Increasing
this value will result in models with linear model
coefficients with lower absolute values.

local_solver : (bool) flag to use the RIDGE regressor from 
the FitSNAP library (default) or to use the sklearn RIDGE
regressor
