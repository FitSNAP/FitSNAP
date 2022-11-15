from matplotlib import pyplot as plt
import numpy as np

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dat = np.loadtxt("pas_comparison.dat")
dat_val = np.loadtxt("pas_comparison_val.dat")

lims = [-2.0, 2.0]
plt.plot(dat[:,0], dat[:,1], 'bo')
plt.plot(dat_val[:,0], dat_val[:,1], 'ro')
plt.plot(lims, lims, 'k-')
plt.legend(["Train", "Validation", "Ideal"])
plt.xlabel("Model electronegativity")
plt.ylabel("Target electronegativity")
plt.xlim(lims[0], lims[1])
plt.ylim(lims[0], lims[1])
plt.savefig("chi_comparison.png", dpi=500)
