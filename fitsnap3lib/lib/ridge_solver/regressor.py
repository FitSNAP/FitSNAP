import numpy as np

# local ridge regressor for regularized fits without sklearn
class Local_Ridge:
    def __init__(self,alpha=1.e-6,fit_intercept=False,mode='SVD'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.mode = mode

    def fit(self,X,Y,mode='SVD'):
        if self.mode == "normal":
            xty = X.T @ Y
            xtx = X.T @ X
            alphI = self.alpha * np.eye(np.shape(xtx)[0])
            normal = (xtx + alphI)
            betas = np.linalg.inv(normal) @ xty
            self.coef_ = betas
        else:
            U, D, V = np.linalg.svd(X,full_matrices=False)
            d2_alph = D**2 + self.alpha
            d2_inv = np.diag(D/d2_alph)
            betas = V.T @ d2_inv @ U.T @ Y
            self.coef_ = betas

    def predict(self,X):
        assert self.coef_ != None, "must have fit before predicting values"
        pred = np.matmul(X,self.coef_)
        return pred
