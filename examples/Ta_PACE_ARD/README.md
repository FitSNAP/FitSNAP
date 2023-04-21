## FitSnap3 Ta example with ACE


## Important Note
As with other ACE fits, regularization is highly
recommended. These Ta potentials are minimally tested and 
are here for demonstration purposes. This example uses ARD
a.k.a relevance vector learning to both apply 
regularization as well as obtain a sparse solution to the 
linear ACE model.

### ARD regression for sparse linear ACE
ARD is the recommended regressor for linear ACE models. 
The use of this method requires the installation of 
sklearn, which can be easily installed with python pip.
In the 29Mar23_ARD example directory, the input for an ARD
fit may be found. This method assumes an elliptical 
gaussian prior for all of the ACE weights. This prior is
adaptively scaled to the variance of the <i>weighted</i>
training data set. This allows for sparse solutions to
the ACE weights, and for the use of fewer descriptors in a
potential. <b>This example requires the installation of
external libraries (sklearn).</b> More info about 
this method can be found in the documentation for sklearn
<a>https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html</a>.


### ARD Hyperparameters

directmethod : (bool) flag to use the hyperparameter scheme
from sklearn, without using training data to help choose
the hyperparameters (default is off)
<br>
<b> Hyperparameters if <u>not</u> using the directmethod </b>
<br>
scap : (float) scaling factor for the prior distribution
of the linear model coefficients (to be rescaled based on
training data variance)
<br>
scai : (float) scaling factor for the prior distribution
of the uncertainty of the linear model coefficients. (to be
rescaled based on training data variance) Simultaneously
increasing or decreasing scap and scai will change the 
absolute scale of model coefficients.
<br>
logcut : (float) log<sub>10</sub> value of the cutoff
threshhold for pruning descriptors. Increasing this value
will add descriptors into the model, lowest uncertainty
first. Decreasing this value will prune descriptors from 
the model, highest uncertainty first.
<br>
<b> Hyperparameters if using the directmethod </b>
<br>
alpha_big : (float) primary parameter for a gamma dist. 
prior for the linear model coefficients
<br>
alpha_small : (float) secondary parameter for the gamma
distribution for the linear model coefficients
<br>
lambda_big : (float) primary parameter for a gamma dist. 
prior for the uncertainties of the linear model 
coefficients.
<br>
lambda_small : (float) secondary parameter for the gamma 
distribution for the uncertainties of the linear model
coefficients.
<br>
threshold_lambda : (int) threshold for pruning descriptors/
(via setting the linear model coefficients to 0)
