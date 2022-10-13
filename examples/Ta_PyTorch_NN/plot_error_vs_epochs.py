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
#plt.rcParams['text.usetex'] = True

dat = np.loadtxt("loss_vs_epochs.dat")

#lims = [-6, 6]
plt.plot(dat[:,0], dat[:,1], 'b-', linewidth=3)
plt.plot(dat[:,0], dat[:,2], 'r-', linewidth=3)
#plt.plot(lims, lims, 'k-')
plt.xlabel("Epochs")
plt.ylabel(r'Loss function')
plt.yscale('log')
plt.legend(["Train", "Validation"])
#plt.xlim(lims[0], lims[1])
#plt.ylim(lims[0], lims[1])
plt.savefig("error_vs_epochs.png", dpi=500)
