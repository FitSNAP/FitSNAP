UQ options:
===========

[SOLVER]
# the standard least-squares fit, but solving the optimization problem instead of SVD or matrix inversions. Can be useful when matrices are ill-conditioned, or when we add regularization
solver = OPT

# analytical result for Bayesian fit, assuming constant noise size (not a great assumption generally)
solver = ANL

# MCMC sampling, currently assuming constant noise size, but unlike the ANL case, there is flexibility if one plays with the log-post function
solver = MCMC

# Fitting with Bayesian compressive sensing, need to learn how to prune bispectrum bases in order for this to be useful. Not working properly yet.
solver = BCS

# if solver==ANL or solver==MCMC, this is the number of fits requested
nsam = 133

[EXTRAS]
plot = 1 # Options are 0, 1, 2. Plots 'diagonal' plots (DFT-vs-SNAP) for all groups, weighted and unweighted, requires matplotlib, and may take time to generate all png files. Option 2 plots with errorbars.





UQ Todos:
=========
Dump the data and write an outside script for simpler UQ debugging
Test BCS and make it prune bases
Pipe to MD propagation
Implement MERR solver (model error), work with GSA for best embedding?
Expose some parameters (e.g. nmcmc) to the input file rather than hardwired
Handle non-positive covariance issues (e.g. when data is truncated)

