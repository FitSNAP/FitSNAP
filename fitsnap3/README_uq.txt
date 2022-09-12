.# this is the exhaustive documentation of UQ options available in main FitSNAP. Not all are guaranteed to provide good results. All UQ solvers output a covariance.npy file in addition to performing the FitSNAP model fit. The current general status is:
ANL solver is the recommended go-to for now. It is fast to run and produces the same fit (within numerical error) as the default SVD solver. The errors calculated from ANL covariance are known to under-predict actual error in general, and biases are likely to exist in datasets that span a range of prediction values.

UQ options:
===========

[SOLVER]
# the standard least-squares fit, but solving the optimization problem instead of SVD or matrix inversions. Can be useful when matrices are ill-conditioned, or when we add regularization
solver = OPT

# analytical result for Bayesian fit, assuming constant noise size (not a great assumption generally)
solver = ANL
nsam = 133            #this is the number of sample fits requested to be drawn from the distribution
cov_nugget = 1.e-10   #this is the small number to be added to the matrix inverse for better conditioning 


# MCMC sampling, currently assuming constant noise size, but unlike the ANL case, there is flexibility if one plays with the log-post function
solver = MCMC
nsam = 133            #this is the number of sample fits requested to be drawn from the distribution
mcmc_num = 1000       #this is the number of total MCMC steps requested
mcmc_gamma = 0.01     #this is the MCMC proposal jump size (smaller gamma increases the acceptance rate)


# Model error embedding approach - powerful but very slow. Requires an optimization that does not run in parallel currently, and is not guaranteed to converge.
solver = MERR
nsam = 133                #this is the number of sample fits requested to be drawn from the distribution
merr_method = iid         #specific liklihood model: options are iid, independent identically distributed, and abc, approximate bayesian computation, and full (too heavy and degenerate, not intended to be used yet)
merr_mult = 0             #0 is additive model error, 1 is multiplicative
merr_cfs = 5 44 3 49 10 33 4 39 38 23       #can provide either a list of coefficient indices to embed on, or "all"
cov_nugget = 1.e-10       #this is the small number to be added to the matrix inverse for better conditioning

# Fitting with Bayesian compressive sensing, need to learn how to prune bispectrum bases in order for this to be useful. Not working properly yet.
solver = BCS

UQ Todos:
=========
Dump the data and write an outside script for simpler UQ debugging
Implement input file handling which coefficient to embed in
Handle non-positive covariance issues (e.g. when data is truncated)
Test BCS and make it prune bases
Save npy not txt in mcmc
Separate out repetitive code for dropping and re-adding zeros into standalone function(s)
Update template and readme

