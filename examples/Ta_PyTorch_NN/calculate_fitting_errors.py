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

def calc_mae_force_magnitude(arr1, arr2):
    """
    Calculate the mean error in force vector difference magnitude.
    This is the average distance between model and target force vectors.
    We do not divide by 3 here, like usually done with MAE force error.
    """
    arr1 = np.reshape(arr1, (int(len(arr1)/3),3))
    arr2 = np.reshape(arr2, (int(len(arr2)/3),3))
    #print(np.shape(arr1))
    diff = arr1-arr2
    norm = np.linalg.norm(diff,axis=1)
    #print(norm)
    mae = np.mean(norm)
    return mae

def calc_mae_force(arr1, arr2):
    abs_diff = np.abs(arr1-arr2) #/natoms
    mae = np.mean(abs_diff)
    return mae

def calc_mae_energy(arr1,arr2):
    abs_diff = np.abs(arr1-arr2) #/natoms
    mae = np.mean(abs_diff)
    return mae

dat = np.loadtxt("force_comparison.dat") # training model | target
dat_val = np.loadtxt("force_comparison_val.dat") # validation model | target
energy = np.loadtxt("energy_comparison.dat") 
energy_val = np.loadtxt("energy_comparison_val.dat")
mae_energy = calc_mae_energy(energy_val[:,0], energy_val[:,1]) #, energy_val[:,2])
print(f"Energy MAE: {mae_energy}")

# calculate distribution of absolute errors on the val set

fmodel = dat_val[:,0]
ftarget = dat_val[:,1]
max_force = np.max(np.abs(ftarget))
lims = [max_force, 1.0]
abs_diff = np.abs(fmodel - ftarget)
max_abs_diff = np.max(abs_diff)
mae_force = calc_mae_force(fmodel, ftarget)
print(f"Force MAE: {mae_force}")
print(f"max abs diff: {max_abs_diff}")
plt.plot(np.abs(ftarget), abs_diff, 'ro', markersize=1)
plt.plot([0,max_force], [mae_force, mae_force], 'k-', markersize=2)
plt.legend(["Test Errors", "MAE"])
plt.xlabel("Test set force component (eV/A)")
plt.ylabel("Absolute error (eV/A)")
plt.xlim(0., max_force)
plt.ylim(0., max_abs_diff)
plt.savefig("force_comparison_distribution.png", dpi=500)
