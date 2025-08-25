import numpy as np

# local ridge regressor for regularized fits without sklearn
class Local_Ridge:
    def __init__(self,alpha=1.e-6,fit_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def fit(self,X,Y):
        # Check for NaN/Inf in inputs and clean if necessary
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"WARNING: NaN/Inf detected in feature matrix X")
            print(f"  Shape: {X.shape}")
            print(f"  NaN count: {np.sum(np.isnan(X))}")
            print(f"  Inf count: {np.sum(np.isinf(X))}")
            # Clean the data
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            print(f"WARNING: NaN/Inf detected in target vector Y")
            print(f"  NaN count: {np.sum(np.isnan(Y))}")
            print(f"  Inf count: {np.sum(np.isinf(Y))}")
            # Clean the data
            Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        xty = np.matmul(X.T,Y)
        xtx = np.matmul(X.T,X)
        
        # Check for numerical issues
        if np.any(np.isnan(xty)) or np.any(np.isinf(xty)):
            print("WARNING: NaN/Inf in X^T Y, cleaning...")
            xty = np.nan_to_num(xty, nan=0.0, posinf=1e10, neginf=-1e10)
            
        if np.any(np.isnan(xtx)) or np.any(np.isinf(xtx)):
            print("WARNING: NaN/Inf in X^T X, cleaning...")
            xtx = np.nan_to_num(xtx, nan=0.0, posinf=1e10, neginf=-1e10)
        
        alphI = self.alpha * np.eye(np.shape(xtx)[0])
        normal = (xtx + alphI)
        
        # Check condition number
        try:
            cond = np.linalg.cond(normal)
            if cond > 1e15:
                print(f"WARNING: Ill-conditioned matrix, condition number = {cond:.2e}")
                # Increase regularization significantly
                alphI = max(1e-4, self.alpha * 1000) * np.eye(np.shape(xtx)[0])
                normal = (xtx + alphI)
        except:
            pass
            
        try:
            betas = np.matmul(np.linalg.inv(normal),xty)
        except np.linalg.LinAlgError:
            print("WARNING: Singular matrix, using pseudo-inverse")
            betas = np.matmul(np.linalg.pinv(normal, rcond=1e-10),xty)
            
        # Final check on coefficients
        if np.any(np.isnan(betas)) or np.any(np.isinf(betas)):
            print("WARNING: NaN/Inf in fitted coefficients, setting problematic values to zero")
            betas = np.nan_to_num(betas, nan=0.0, posinf=0.0, neginf=0.0)
            
        self.coef_ = betas

    def predict(self,X):
        assert self.coef_ is not None, "must have fit before predicting values"
        
        # Check input
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("WARNING: NaN/Inf in prediction matrix, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
        pred = np.matmul(X,self.coef_)
        
        # Check output
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            print("WARNING: NaN/Inf in predictions, cleaning...")
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
            
        return pred
