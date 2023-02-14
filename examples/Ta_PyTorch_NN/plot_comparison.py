"""
Load detailed fitting data from FitSNAP NN outputs into a dataframe and process/plot the results.

Usage:

    python plot_comparison.py
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fitsnap3lib.tools.nn_tools import NNTools

# Create NNTools object with output files to analyze.
nnt = NNTools("peratom.dat", "perconfig.dat")

# Calculate energy and force errors.
quantities = ["Force", "Energy"]
for q in quantities:
    e_err = nnt.calc_errors(q) # E.g. e_err['test']['mae'] gives test MAE
    for testkey in e_err.keys():
        for errkey in e_err[testkey].keys():
            print(f"  {q} {testkey} {errkey}: {e_err[testkey][errkey]}")

# Plot energy and force comparisons.
for q in quantities:
    nnt.plot_comparisons(q)