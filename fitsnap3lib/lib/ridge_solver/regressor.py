import numpy as np

# local ridge regressor for regularized fits without sklearn
class Local_Ridge:
    def __init__(self,alpha=1.e-6,fit_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self,X,Y):
        xty = np.matmul(X.T,Y)
        xtx = np.matmul(X.T,X)
        alphI = self.alpha * np.eye(np.shape(xtx)[0])
        normal = (xtx + alphI)
        betas = np.matmul(np.linalg.inv(normal),xty)
        self.coef_ = betas

    def predict(self,X):
        assert self.coef_ != None, "must have fit before predicting values"
        pred = np.matmul(X,self.coef_)
        return pred
