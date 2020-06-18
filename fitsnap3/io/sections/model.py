from fitsnap3.io.sections.sections import Section


class Model(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.chemflag = self.get_value("MODEL", "chemflag", "0", "bool")
        self.wselfallflag = self.get_value("MODEL", "wselfallflag", "0", "bool")
        self.bzeroflag = self.get_value("MODEL", "bzeroflag", "0", "bool")
        self.quadraticflag = self.get_value("MODEL", "quadraticflag", "0", "bool")
        # self.solver = self.get_value("MODEL", "solver", "SVD")
        # self.normalweight = self.get_value("MODEL", "normalweight", "-12", "float")
        # self.normratio = self.get_value("MODEL", "normratio", "0.5", "float")
        # self.compute_dbvb = self.get_value("MODEL", "compute_dbvb", "0", "bool")
        # self.compute_testerrs = self.get_value("MODEL", "compute_testerrs", "0", "bool")
        # self.delete()
