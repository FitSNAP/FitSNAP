import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def compute_stdev(a, cov, fit_sam, method="chol"):
    if method == "sam":
        pf_stdev = np.std(fit_sam @ a.T, axis=0)
    elif method == "chol":
        assert(cov is not None)
        chol = np.linalg.cholesky(cov)
        mat = a @ chol
        pf_stdev = np.linalg.norm(mat, axis=1)
    elif method == "choleye":
        assert(cov is not None)
        eigvals = np.linalg.eigvalsh(cov)
        chol = np.linalg.cholesky(cov+(abs(eigvals[0]) + 1e-14) * np.eye(cov.shape[0]))
        mat = a @ chol
        pf_stdev = np.linalg.norm(mat, axis=1)
    elif method == "svd":
        assert(cov is not None)
        u, s, vh = np.linalg.svd(cov, hermitian=True)
        mat = (a @ u) @ np.sqrt(np.diag(s))
        pf_stdev = np.linalg.norm(mat, axis=1)
    elif method == "loop":
        assert(cov is not None)
        tmp = np.dot(a, cov)
        pf_stdev = np.empty(a.shape[0])
        for ipt in range(a.shape[0]):
            pf_stdev[ipt] = np.sqrt(np.dot(tmp[ipt, :], a[ipt, :]))
    elif method == "fullcov":
        assert(cov is not None)
        pf_stdev = np.sqrt(np.diag((a @ cov) @ a.T))
    else:
        pf_stdev = np.zeros(a.shape[0])
    return pf_stdev

def errors(group, rtype, indices, df, weight_flag, covariance, fit_sam):
    this_true, this_pred = df['truths'][indices], df['preds'][indices]
    A_matrix_columns = [num for num in df.columns if isinstance(num, (int))]
    A = df[A_matrix_columns].to_numpy()
    this_a = df.iloc[:, 0:A.shape[1]].loc[indices].to_numpy()
    if weight_flag == 'Weighted':
        w = df['weights'].array[indices]
        this_true, this_pred = w * this_true, w * this_pred
        nconfig = np.count_nonzero(w)
    else:
        nconfig = len(this_pred)
    res = this_true - this_pred
    mae = np.sum(np.abs(res) / nconfig)
    ssr = np.square(res).sum()
    mse = ssr / nconfig
    rmse = np.sqrt(mse)
    rsq = 1 - ssr / np.sum(np.square(this_true - (this_true / nconfig).sum()))
    error_record = {
        "Group": group,
        "Weighting": weight_flag,
        "Subsystem": rtype,
        "ncount": nconfig,
        "mae": mae,
        "rmse": rmse,
        "rsq": rsq,
        "residual": res
    }

#    self.errors.append(error_record)
    if covariance is None:
        if fit_sam is not None:
            pf_std = compute_stdev(this_a, covariance, fit_sam, "sam")
        else:
            pf_std = np.zeros(this_a.shape[0])
    else:
        try:
            pf_std = compute_stdev(this_a, covariance, fit_sam, "chol")
        except np.linalg.LinAlgError:
            pf_std = compute_stdev(this_a, covariance, fit_sam, "svd")
    
    mpl.rc('axes', linewidth=2, grid=True, labelsize=16)
    mpl.rc('figure', max_open_warning=500)

    plt.figure(figsize=(8,8))

#    if config.sections["EXTRAS"].plot == 1:
#        plt.plot(this_true, this_pred, 'ro', markersize=11, markeredgecolor='w')
#    elif config.sections["EXTRAS"].plot > 1:
    plt.errorbar(this_true, this_pred, yerr=pf_std, fmt = 'ro', ecolor='r', elinewidth=2, markersize=11, markeredgecolor='w')
            
    xmin, xmax = min(np.min(this_true),np.min(this_pred)), max(np.max(this_true),np.max(this_pred))
    plt.plot([xmin, xmax], [xmin, xmax], 'k--', linewidth=1)
    plt.xlabel('DFT')
    plt.ylabel('SNAP')
    plt.title(group+'; '+rtype+'; '+weight_flag)
    plt.gcf().tight_layout()
    plt.savefig('dm_'+group+'_'+rtype+'_'+weight_flag+'.png')
    plt.clf()



df = pd.read_pickle("FitSNAP.df")
try:
    fit_mean = np.load("mean.npy")
except:
    fit_mean = None
try:
    fit_cov = np.load("covariance.npy")
except:
    fit_cov = None
#fit_sam = np.random.multivariate_normal(fit_mean, fit_cov, size=(nsam,)) if you want to use sampling
fit_sam = None

Row_Types = df['Row_Type'].unique()
groups = df['Groups'].unique()
for option in ["Unweighted", "Weighted"]:
    weight_flag = option
    for r_type in Row_Types:
        errors("*ALL", r_type, (df['Row_Type'] == r_type), df, weight_flag, fit_cov, fit_sam)
        for group in groups:
            group_filter = df['Groups'] == group
            errors(group, r_type, group_filter & (df['Row_Type'] == r_type), df, weight_flag, fit_cov, fit_sam)
    


