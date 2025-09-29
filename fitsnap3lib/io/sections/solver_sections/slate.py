from fitsnap3lib.io.sections.sections import Section


class Slate(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha', 'method', 'directmethod', 'scap', 'scai', 'logcut', 'max_iterations', 'tolerance']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SLATE")

        self.alpha = self.get_value("SLATE", "alpha", "1.0E-8", "float")
        
        # Method selection: RIDGE (default) or ARD
        self.method = self.get_value("SLATE", "method", "RIDGE", "str")
        
        # ARD parameters (only used when method=ARD)
        self.directmethod = self.get_value("SLATE", "directmethod", "0", "int")
        self.scap = self.get_value("SLATE", "scap", "1.E-4", "float")
        self.scai = self.get_value("SLATE", "scai", "1.E-4", "float")
        self.logcut = self.get_value("SLATE", "logcut", "0.3", "float")
        self.max_iterations = self.get_value("SLATE", "max_iterations", "100", "int")
        self.tolerance = self.get_value("SLATE", "tolerance", "1.0E-3", "float")
        
        self.delete()
