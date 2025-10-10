from fitsnap3lib.io.sections.sections import Section


class Slate(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['method', 'alpha',
            'max_iter', 'tol', 'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'threshold_lambda']
                       
        self._check_section()
        self._check_if_used("SOLVER", "solver", "SLATE")
        
        # Method selection: RIDGE (default) or ARD
        self.method = self.get_value("SLATE", "method", "RIDGE", "str")

        # alpha for RIDGE
        self.alpha = self.get_value("SLATE", "alpha", "1.0E-8", "float")

        # Maximum number of iterations
        self.max_iter = self.get_value("SLATE", "max_iter", "300", "int")
        
        # Stop the algorithm if w has converged
        self.tol = self.get_value("SLATE", "tol", "1e-3", "float")
        
        # Shape and inverse scale (rate) parameters for Gamma prior over alpha
        self.alpha_1 = self.get_value("SLATE", "alpha_1", "1e-6", "float")
        self.alpha_2 = self.get_value("SLATE", "alpha_2", "1e-6", "float")

        # Shape and inverse scale (rate) parameters for Gamma prior over lambda
        self.lambda_1 = self.get_value("SLATE", "lambda_1", "1e-6", "float")
        self.lambda_2 = self.get_value("SLATE", "lambda_2", "1e-6", "float")

        # Threshold for removing (pruning) weights with high precision from the computation.
        self.threshold_lambda = self.get_value("SLATE", "threshold_lambda", "100", "float")

        # Whether to calculate the intercept for this model. If set to false,
        # no intercept will be used in calculations (ie. data is expected to be centered)
        # fit_intercept
        # FIXME: related to bzeroflag
        
        self.delete()
