from fitsnap3lib.io.sections.sections import Section


class Lasso(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha', 'max_iter']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alpha = self.get_value("LASSO", "alpha", "1.0E-8", "float")
        self.max_iter = self.get_value("LASSO", "max_iter", "2000", "int")
        self.delete()
