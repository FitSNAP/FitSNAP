from fitsnap3lib.io.sections.sections import Section


class Ard(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alphabig', 'alphasmall', 'lambdabig', 'lambdasmall', 'threshold_lambda','directmethod','scap','scai','logcut']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alphabig = self.get_value("ARD", "alphabig", "1.0E-12", "float")
        self.alphasmall = self.get_value("ARD", "alphasmall", "1.0E-14", "float")
        self.lambdabig = self.get_value("ARD", "lambdabig", "1.0E-6", "float")
        self.lambdasmall = self.get_value("ARD", "lambdasmall", "1.0E-6", "float")
        self.threshold_lambda = self.get_value("ARD", "threshold_lambda", "100000", "int")
        self.directmethod = self.get_value("ARD", "directmethod", "0", "int")
        self.scap = self.get_value("ARD", "scap", "1.e-3", "float")
        self.scai = self.get_value("ARD", "scai", "1.e-3", "float")
        self.logcut = self.get_value("ARD", "logcut", "0.3", "float")

        self.delete()
