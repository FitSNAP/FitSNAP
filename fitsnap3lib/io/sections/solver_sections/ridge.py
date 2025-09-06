from fitsnap3lib.io.sections.sections import Section


class Ridge(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha','local_solver','tile_size']
        self._check_section()

        # Check if solver is either RIDGE or RidgeSlate
        solver_type = self.get_value("SOLVER", "solver", "SVD")
        if solver_type.upper() not in ["RIDGE", "RIDGESLATE"]:
            raise UserWarning("{} solver section is in input, but solver is set to {}".format(self.name, solver_type))

        self.alpha = self.get_value("RIDGE", "alpha", "1.0E-8", "float")
        self.local_solver = self.get_value("RIDGE", "local_solver", "0", "bool")
        self.tile_size = self.get_value("RIDGE", "tile_size", "256", "int")  # For RidgeSlate
        self.delete()
