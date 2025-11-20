from fitsnap3lib.io.sections.sections import Section


class Seq_Ace_Lasso(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha', 'max_iter', 'enforce_pos_rank']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        #self.alpha = self.get_value("SEQ_ACE_LASSO", "alpha", "1.0E-8", "float")
        self.alpha = self.get_value("SEQ_ACE_LASSO","alpha","1.0E-8").split()
        self.max_iter = self.get_value("SEQ_ACE_LASSO", "max_iter", 10000, "int")
        self.enforce_pos_rank = self.get_value("SEQ_ACE_LASSO", "enforce_pos_rank", "0").split()
        self.delete()
