from fitsnap3lib.io.sections.sections import Section


class Slate(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['method', 'alpha',
            'max_iter', 'tol', 
            'alphabig', 'alphasmall', 'lambdabig', 'lambdasmall', 
            'threshold_lambda', 'threshold_gamma', 'pruning_method',
            'directmethod', 'scap', 'scai', 'logcut']
                       
        self._check_section()
        self._check_if_used("SOLVER", "solver", "SLATE")
        
        # Method selection: RIDGE (default) or ARD
        self.method = self.get_value("SLATE", "method", "RIDGE", "str")

        # alpha for RIDGE
        self.alpha = self.get_value("SLATE", "alpha", "1.0E-8", "float")

        # ARD parameters - matching legacy ARD section
        # Maximum number of iterations
        self.max_iter = self.get_value("SLATE", "max_iter", "300", "int")
        
        # Stop the algorithm if w has converged
        self.tol = self.get_value("SLATE", "tol", "1e-3", "float")
        
        # Direct method hyperparameters (used if directmethod=1)
        self.alphabig = self.get_value("SLATE", "alphabig", "1.0E-12", "float")
        self.alphasmall = self.get_value("SLATE", "alphasmall", "1.0E-14", "float")
        self.lambdabig = self.get_value("SLATE", "lambdabig", "1.0E-6", "float")
        self.lambdasmall = self.get_value("SLATE", "lambdasmall", "1.0E-6", "float")
        
        # Pruning method: 'lambda' (default) or 'gamma'
        self.pruning_method = self.get_value("SLATE", "pruning_method", "lambda", "str")
        
        # Lambda threshold for removing (pruning) weights with high precision from the computation.
        # If not specified, will be auto-computed as 10^(int(abs(log10(ap))) + logcut)
        self.threshold_lambda = self.get_value("SLATE", "threshold_lambda", "0", "float")
        
        # Gamma threshold: fraction of parameter usage (0-1 range)
        # Features with gamma < threshold_gamma are pruned
        # More interpretable than lambda threshold
        self.threshold_gamma = self.get_value("SLATE", "threshold_gamma", "0.1", "float")
        
        # Adaptive hyperparameter mode (0=adaptive using scap/scai, 1=direct using alphabig/lambdasmall)
        self.directmethod = self.get_value("SLATE", "directmethod", "0", "int")
        
        # Scaling factors for adaptive hyperparameters (used if directmethod=0)
        self.scap = self.get_value("SLATE", "scap", "1.e-3", "float")
        self.scai = self.get_value("SLATE", "scai", "1.e-3", "float")
        
        # Log cutoff for auto-computing threshold_lambda (used if threshold_lambda not specified)
        self.logcut = self.get_value("SLATE", "logcut", "0.3", "float")

        # Whether to calculate the intercept for this model. If set to false,
        # no intercept will be used in calculations (ie. data is expected to be centered)
        # fit_intercept
        # FIXME: related to bzeroflag
        
        self.delete()
