.
UQ options:
===========

[SOLVER]
# the standard least-squares fit, but solving the optimization problem instead of SVD or matrix inversions. Can be useful when matrices are ill-conditioned, or when we add regularization
solver = OPT

# analytical result for Bayesian fit, assuming constant noise size (not a great assumption generally)
solver = ANL

# MCMC sampling, currently assuming constant noise size, but unlike the ANL case, there is flexibility if one plays with the log-post function
solver = MCMC

# Model error embedding approach
solver = MERR

# Fitting with Bayesian compressive sensing, need to learn how to prune bispectrum bases in order for this to be useful. Not working properly yet.
solver = BCS

# if solver==ANL or solver==MCMC or solver==MERR, this is the number of fits requested
nsam = 133

# if solver==ANL, or solver==MERR this is the small number to be added to inverse covariance for better conditioning
cov_nugget = 1.e-10

# if solver==MCMC, this is the number of total MCMC steps requested
mcmc_num = 1000

# if solver==MCMC, this is the MCMC proposal jump size (smaller gamma increases the acceptance rate)
mcmc_gamma = 0.01

# if solver==MERR, this is the specific likelihood method for model error
merr_method = abc 
# Options are abc (Approximate Bayesian Computation), iid (independent identically distributed approximation), or full (too heavy and degenerate, not intended to be used yet)

# if solver==MERR, this controls whether we want multiplicative or additive embedding
merr_mult = 1   
# Options are 0 (additive) or 1 (multiplicative)

# if solver==MERR, this is a list of integers for coefficients where model error is embedded
merr_cfs = 2 5 0
# Options are a string of space-separated integers, or a string 'all' (embed in all bispectum coefficients). The option 'all' leads to a high-dimensional inference with less than robust results.

[EXTRAS] 
plot = 1 
# Options are 0, 1, 2. Plots 'diagonal' plots (DFT-vs-SNAP) for all groups, weighted and unweighted, requires matplotlib, and may take time to generate all png files. Option 2 plots with errorbars.





UQ Todos:
=========
Dump the data and write an outside script for simpler UQ debugging
Implement input file handling which coefficient to embed in
Handle non-positive covariance issues (e.g. when data is truncated)
Test BCS and make it prune bases
Save npy not txt in mcmc
Implement Logan's anl additions
Update template and readme

