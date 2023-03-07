import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

class NNTools():
    """
    Class containing methods that help process output for NN models.

    Args:
        peratom_file : Path to a per atom data file from FitSNAP NN output. Defaults to None.
        perconfig_file : Path to a per config data file from FitSNAP NN output. Defaults to None.

    Attributes:
        dfa : Dataframe of per atom data.
        dfc : Dataframe of per config data.

    """
    def __init__(self, peratom_file: str = None, perconfig_file: str = None):
        self.dfa = pd.read_csv(peratom_file,
                   delimiter = ' ',
                   lineterminator = '\n',
                   header = 0) if peratom_file else None
        self.dfc = pd.read_csv(perconfig_file,
                   delimiter = ' ',
                   lineterminator = '\n',
                   header = 0) if peratom_file else None

    def calc_errors(self, quantity: str) -> dict:
        """
        Calculate errors using FitSNAP NN output files.

        Args:
            quantity : "Energy" or "Force".

        Returns:
            Tuple of error values (mae_train, mae_test, rmse_train, rmse_test) for given a quantity.
            Dictionary of error values for given quantity. First key is error metric, second key is 
            train/test. E.g. ["test"]["mae"] denotes the test MAE for a given quantity.
        """

        ret = {}
        ret["train"] = {}
        ret["test"] = {}
        if (quantity == "Energy"):
            test_bool = self.dfc.loc[:, "Testing_Bool"].tolist()
            test_idx = [i for i, x in enumerate(test_bool) if x]
            train_idx = [i for i, x in enumerate(test_bool) if not x]
            # Extract train energies.
            true_train = np.array(self.dfc.loc[train_idx, "Energy_Truth"].tolist())
            pred_train = np.array(self.dfc.loc[train_idx, "Energy_Pred"].tolist())
            ret["train"]["mae"] = np.mean(abs(true_train - pred_train))
            ret["train"]["rmse"] = np.sqrt(np.mean(np.square(true_train - pred_train)))
            # Extract test energies.
            true_test = np.array(self.dfc.loc[test_idx, "Energy_Truth"].tolist())
            pred_test = np.array(self.dfc.loc[test_idx, "Energy_Pred"].tolist())
            ret["test"]["mae"] = np.mean(abs(true_test - pred_test))
            ret["test"]["rmse"] = np.sqrt(np.mean(np.square(true_test - pred_test)))
        elif (quantity == "Force"):
            test_bool = self.dfa.loc[:, "Testing_Bool"].tolist()
            test_idx = [i for i, x in enumerate(test_bool) if x]
            train_idx = [i for i, x in enumerate(test_bool) if not x]
            # Extract train forces.
            true_train = np.array(self.dfa.loc[train_idx, "Fx_Truth"].tolist() + \
                                  self.dfa.loc[train_idx, "Fy_Truth"].tolist() + \
                                  self.dfa.loc[train_idx, "Fz_Truth"].tolist())
            pred_train = np.array(self.dfa.loc[train_idx, "Fx_Pred"].tolist() + \
                                  self.dfa.loc[train_idx, "Fy_Pred"].tolist() + \
                                  self.dfa.loc[train_idx, "Fz_Pred"].tolist())
            ret["train"]["mae"] = np.mean(abs(true_train - pred_train))
            ret["train"]["rmse"] = np.sqrt(np.mean(np.square(true_train - pred_train)))
            # Extract test forces.
            true_test = np.array(self.dfa.loc[test_idx, "Fx_Truth"].tolist() + \
                                 self.dfa.loc[test_idx, "Fy_Truth"].tolist() + \
                                 self.dfa.loc[test_idx, "Fz_Truth"].tolist())
            pred_test = np.array(self.dfa.loc[test_idx, "Fx_Pred"].tolist() + \
                                 self.dfa.loc[test_idx, "Fy_Pred"].tolist() + \
                                 self.dfa.loc[test_idx, "Fz_Pred"].tolist())
            ret["test"]["mae"] = np.mean(abs(true_test - pred_test))
            ret["test"]["rmse"] = np.sqrt(np.mean(np.square(true_test - pred_test)))
        else:
            raise Exception(f"{quantity} should be either 'Force' or 'Energy'")

        return ret

    def plot_comparisons(self, quantity: str, mode="Linear") -> dict:
        """
        Plot comparisons between truth/prediction energies or forces.

        Args:
            quantity : "Energy" or "Force".
            mode : "Distribution" or "Linear".
        """

        if (quantity == "Energy"):
            test_bool = self.dfc.loc[:, "Testing_Bool"].tolist()
            test_idx = [i for i, x in enumerate(test_bool) if x]
            train_idx = [i for i, x in enumerate(test_bool) if not x]
            # Extract train energies.
            true_train = np.array(self.dfc.loc[train_idx, "Energy_Truth"].tolist())
            pred_train = np.array(self.dfc.loc[train_idx, "Energy_Pred"].tolist())
            # Extract test energies.
            true_test = np.array(self.dfc.loc[test_idx, "Energy_Truth"].tolist())
            pred_test = np.array(self.dfc.loc[test_idx, "Energy_Pred"].tolist())
            # Axes labels
            if (mode=="Linear"):
                xlabel = "Model energy (eV/atom)"
                ylabel = "Target energy (eV/atom)"
            elif (mode=="Distribution"):
                xlabel = "Abs. target energy (eV/atom)"
                ylabel = "Abs. error (eV/atom)"
            filename = "energy_comparison.png"
        elif (quantity == "Force"):
            test_bool = self.dfa.loc[:, "Testing_Bool"].tolist()
            test_idx = [i for i, x in enumerate(test_bool) if x]
            train_idx = [i for i, x in enumerate(test_bool) if not x]
            # Extract train forces
            true_train = np.array(self.dfa.loc[train_idx, "Fx_Truth"].tolist() + \
                                  self.dfa.loc[train_idx, "Fy_Truth"].tolist() + \
                                  self.dfa.loc[train_idx, "Fz_Truth"].tolist())
            pred_train = np.array(self.dfa.loc[train_idx, "Fx_Pred"].tolist() + \
                                  self.dfa.loc[train_idx, "Fy_Pred"].tolist() + \
                                  self.dfa.loc[train_idx, "Fz_Pred"].tolist())
            # Extract test forces
            true_test = np.array(self.dfa.loc[test_idx, "Fx_Truth"].tolist() + \
                                 self.dfa.loc[test_idx, "Fy_Truth"].tolist() + \
                                 self.dfa.loc[test_idx, "Fz_Truth"].tolist())
            pred_test = np.array(self.dfa.loc[test_idx, "Fx_Pred"].tolist() + \
                                 self.dfa.loc[test_idx, "Fy_Pred"].tolist() + \
                                 self.dfa.loc[test_idx, "Fz_Pred"].tolist())
            # Axes labels
            if (mode=="Linear"):
                xlabel = r"Model force component (eV/$\AA$)"
                ylabel = r"Target force component (eV/$\AA$)"
            elif (mode=="Distribution"):
                xlabel = r"Abs. target force component (eV/$\AA$)"
                ylabel = r"Abs. error (eV/$\AA$)"
            filename = "force_comparison.png"
        else:
            raise Exception(f"{quantity} should be either 'Force' or 'Energy'")


        if (mode=="Linear"):
            lims = [min(true_train), max(true_train)]
            plt.plot(pred_train, true_train, 'bo', markersize=5)
            plt.plot(pred_test, true_test, 'ro', markersize=5)
            plt.plot(lims, lims, 'k-')
            plt.legend(["Train", "Validation", "Ideal"])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(lims[0], lims[1])
            plt.ylim(lims[0], lims[1])
            plt.savefig(filename, dpi=500)
            plt.clf()
        elif (mode=="Distribution"):
            # Plot absolute error vs. target magnitude.
            plt.plot(abs(true_train), abs(true_train-pred_train), 'bo', markersize=5)
            plt.plot(abs(true_test), abs(true_test-pred_test), 'ro', markersize=5)
            plt.legend(["Train", "Validation"])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(filename, dpi=500)
            plt.clf()
