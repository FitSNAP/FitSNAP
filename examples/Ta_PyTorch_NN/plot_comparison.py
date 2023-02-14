"""
Load detailed fitting data from FitSNAP NN outputs into a dataframe and process/plot the results.

Usage:

    python plot_comparison.py
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

# Load per atom dataframe.
dfa = pd.read_csv("peratom.dat",
                  delimiter = ' ',
                  lineterminator = '\n',
                  header = 0)
# Load per config dataframe.
dfc = pd.read_csv("perconfig.dat",
                  delimiter = ' ',
                  lineterminator = '\n',
                  header = 0)

# Find train/test rows for energies.
test_bool = dfc.loc[:, "Testing_Bool"].tolist()
test_idx = [i for i, x in enumerate(test_bool) if x]
train_idx = [i for i, x in enumerate(test_bool) if not x]
# Extract train energies.
e_true_train = np.array(dfc.loc[train_idx, "Energy_Truth"].tolist())
e_pred_train = np.array(dfc.loc[train_idx, "Energy_Pred"].tolist())
e_mae_train = np.mean(abs(e_true_train - e_pred_train))
e_rmse_train = np.sqrt(np.mean(np.square(e_true_train - e_pred_train)))
# Extract test energies.
e_true_test = np.array(dfc.loc[test_idx, "Energy_Truth"].tolist())
e_pred_test = np.array(dfc.loc[test_idx, "Energy_Pred"].tolist())
e_mae_test = np.mean(abs(e_true_test- e_pred_test))
e_rmse_test = np.sqrt(np.mean(np.square(e_true_test - e_pred_test)))

# Find train/test rows for forces.
test_bool = dfa.loc[:, "Testing_Bool"].tolist()
test_idx = [i for i, x in enumerate(test_bool) if x]
train_idx = [i for i, x in enumerate(test_bool) if not x]
# Extract train forces.
f_true_train = np.array(dfa.loc[train_idx, "Fx_Truth"].tolist() + \
                        dfa.loc[train_idx, "Fy_Truth"].tolist() + \
                        dfa.loc[train_idx, "Fz_Truth"].tolist())
f_pred_train = np.array(dfa.loc[train_idx, "Fx_Pred"].tolist() + \
                        dfa.loc[train_idx, "Fy_Pred"].tolist() + \
                        dfa.loc[train_idx, "Fz_Pred"].tolist())
f_mae_train = np.mean(abs(f_true_train - f_pred_train))
f_rmse_train = np.sqrt(np.mean(np.square(f_true_train - f_pred_train)))
# Extract test forces.
f_true_test = np.array(dfa.loc[test_idx, "Fx_Truth"].tolist() + \
                       dfa.loc[test_idx, "Fy_Truth"].tolist() + \
                       dfa.loc[test_idx, "Fz_Truth"].tolist())
f_pred_test = np.array(dfa.loc[test_idx, "Fx_Pred"].tolist() + \
                       dfa.loc[test_idx, "Fy_Pred"].tolist() + \
                       dfa.loc[test_idx, "Fz_Pred"].tolist())
f_mae_test = np.mean(abs(f_true_test - f_pred_test))
f_rmse_test = np.sqrt(np.mean(np.square(f_true_test - f_pred_test)))

print(f"Force Train MAE: {f_mae_train}")
print(f"Force Train RMSE: {f_rmse_train}")
print(f"Force Test MAE: {f_mae_test}")
print(f"Force Test RMSE: {f_rmse_test}")
print(f"Energy Train MAE: {e_mae_train}")
print(f"Energy Train RMSE: {e_rmse_train}")
print(f"Energy Test MAE: {e_mae_test}")
print(f"Energy Test RMSE: {e_rmse_test}")



dat = np.loadtxt("force_comparison.dat")
dat_val = np.loadtxt("force_comparison_val.dat")

# Plot energy comparison.
xlims = [min(e_pred_train), max(e_pred_train)]
ylims = [min(e_true_train), max(e_true_train)]
print(e_pred_train)
print(e_true_train)
plt.plot(e_pred_train, e_true_train, 'bo', markersize=1)
plt.plot(e_pred_test, e_true_test, 'ro', markersize=2)
plt.plot(ylims, ylims, 'k-')
plt.legend(["Train", "Validation", "Ideal"])
plt.xlabel("Model energy (eV/atom)")
plt.ylabel("Target energy (eV/atom)")
plt.xlim(xlims[0], xlims[1])
plt.ylim(ylims[0], ylims[1])
plt.savefig("energy_comparison.png", dpi=500)

plt.clf()

# Plot force comparison.
lims = [min(f_true_train), max(f_true_train)]
plt.plot(f_pred_train, f_true_train, 'bo', markersize=1)
plt.plot(f_pred_test, f_true_test, 'ro', markersize=2)
plt.plot(lims, lims, 'k-')
plt.legend(["Train", "Validation", "Ideal"])
plt.xlabel("Model force component (eV/A)")
plt.ylabel("Target force component (eV/A)")
plt.xlim(lims[0], lims[1])
plt.ylim(lims[0], lims[1])
plt.savefig("force_comparison.png", dpi=500)
