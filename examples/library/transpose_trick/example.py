"""
Python script for performing the transpose trick C = A^T * A and d = A^T * b, summing these 
matrices for all configurations, then performing the fit.

Usage:

    python example.py

Afterwards, use the `in.run` LAMMPS script to run MD.

NOTE: See below for info on which variables to change for different options.
"""

from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import copy
import numpy as np
from scipy.linalg import lstsq
from sys import float_info as fi
from fitsnap3lib.lib.ridge_solver.regressor import Local_Ridge
from sklearn.linear_model import Ridge

def fit(c, d):
    coeffs, residues, rank, s = lstsq(c, d, 1.0e-13)
    print(coeffs)
    print(np.shape(coeffs))
    #assert(False)

def ridge(c, d):

    alval = 1.e-6
    #reg = Local_Ridge(alpha = alval, fit_intercept = False)
    reg = Ridge(alpha = alval, fit_intercept = False)
    reg.fit(c, d)
    print(reg.coef_)
    np.savetxt("coeffs.txt", reg.coef_.T)
    #assert(False)
    return reg.coef_.T # use this if use sklearn ridge
    #return reg.coef_[:,np.newaxis]



# Declare a communicator (this can be a custom communicator as well).
comm = MPI.COMM_WORLD

# Create an input dictionary containing settings.
settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"SCRAPER":
    {
    "scraper": "JSON" 
    },
"PATH":
    {
    "dataPath": "../../Ta_Linear_JCP2014/JSON"
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 1,
    "dump_truth": 1,
    "dump_weights": 1,
    "dump_dataframe": 1
    },
"GROUPS":
    {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_BCC" :  "1.0    0.0       100             1               1.00E-08",
    "Displaced_FCC" :  "1.0    0.0       100             1               1.00E-08",
    "Elastic_BCC"   :  "1.0    0.0     1.00E-08        1.00E-08        0.0001",
    "Elastic_FCC"   :  "1.0    0.0     1.00E-09        1.00E-09        1.00E-09",
    "GSF_110"       :  "1.0    0.0      100             1               1.00E-08",
    "GSF_112"       :  "1.0    0.0      100             1               1.00E-08",
    "Liquid"        :  "1.0    0.0       4.67E+02        1               1.00E-08",
    "Surface"       :  "1.0    0.0       100             1               1.00E-08",
    "Volume_A15"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_BCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09",
    "Volume_FCC"    :  "1.0    0.0      1.00E+00        1.00E-09        1.00E-09"
    },
"MEMORY":
    {
    "override": 0
    }
}

# Alternatively, settings could be provided in a traditional input file:
#settings = "../../Ta_Linear_JCP2014/Ta-example.in"
#settings = "../../Ta_PACE/Ta.in"
settings = "../../Ta_PACE_RIDGE/Ta.in"

# Create a FitSnap instance using the communicator and settings:
fitsnap = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Scrape configurations to create and populate the `snap.data` list of dictionaries with structural info.
fitsnap.scrape_configs()
# Calculate descriptors for all structures in the `snap.data` list.
# This is performed in parallel over all processors in `comm`.
# Descriptor data is stored in the shared arrays.
print(len(fitsnap.data))
data = copy.deepcopy(fitsnap.data)
print(len(data))

# Loop over each configuration in data 
a_width = fitsnap.calculator.get_width()
c = np.zeros((a_width,a_width)) # This will also include weights.
d = np.zeros((a_width,1))
for m, config in enumerate(data):
    print(m)
    fitsnap.data = [config]
    fitsnap.process_configs()
    # See what original A matrix looks like:
    #print(fitsnap.pt.shared_arrays['a'].array)
    #print(np.shape(fitsnap.pt.shared_arrays['a'].array))
    #print(fitsnap.calculator.a)
    #fit(fitsnap.calculator.a, fitsnap.calculator.w, fitsnap.calculator.b)
    # Make aw and bw
    #w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]
    a = fitsnap.calculator.a
    b = fitsnap.calculator.b
    w = fitsnap.calculator.w
    aw, bw = w[:, np.newaxis] * a, w * b
    if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
        pass
    else:
        print("ill conditioned for transpose trick")
        #print(a)
        #print(fitsnap.calculator.a)
        #assert(np.all(a == fitsnap.calculator.a))
        #assert(False)
    cm = np.matmul(np.transpose(aw), aw)
    dm = np.matmul(np.transpose(aw), bw[:,np.newaxis])
    c += cm
    d += dm
    #assert(False)

fit(c,d)
coeffs = ridge(c,d)

fitsnap.data = data

# Process configs to get entire A matrix.
fitsnap.process_configs()

# Use coeffs
fitsnap.fit = coeffs
fitsnap.solver.fit = coeffs

fitsnap.solver.fit_gather()
fitsnap.solver.error_analysis()

fitsnap.output.output(coeffs, fitsnap.solver.errors)

# Output coeffs here (will error due to solver.errors being None)
#fitsnap.output.output(coeffs, fitsnap.solver.errors)


"""
fitsnap.process_configs()
# Good practice after a large parallel operation is to impose a barrier to wait for all procs to complete.
fitsnap.pt.all_barrier()
# Perform a fit using data in the shared arrays.
fitsnap.perform_fit()
fitsnap.write_output()
"""
