from fitsnap3lib.io.sections.sections import Section


class Ridge(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha','local_solver']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alpha = self.get_value("RIDGE", "alpha", "1.0E-8", "float")
        self.local_solver = self.get_value("RIDGE", "local_solver", "1", "bool")
        self.delete()
