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

mae = 0.
mae_val = 0.
for i in range(1,5+1):
    dat = np.loadtxt(f"force_comparison_{i}.dat")
    dat_val = np.loadtxt(f"force_comparison_val_{i}.dat")
    mae += np.mean(np.abs(dat[:,0]-dat[:,1]))
    mae_val += np.mean(np.abs(dat_val[:,0]-dat_val[:,1]))
    print(f"{mae} {mae_val}")
mae = mae/5.
mae_val = mae_val/5.

print(f"{mae} {mae_val}")
