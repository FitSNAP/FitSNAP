from fitsnap3lib.io.sections.sections import Section


class Seq_Ace_Ridge(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha','local_solver','enforce_pos_rank']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        #self.alpha = self.get_value("SEQ_ACE_RIDGE", "alpha", "1.0E-8", "float")
        self.alpha = self.get_value("SEQ_ACE_RIDGE","alpha","1.0E-8").split()
        self.local_solver = self.get_value("SEQ_ACE_RIDGE", "local_solver", "1", "bool")
        self.enforce_pos_rank = self.get_value("SEQ_ACE_RIDGE", "enforce_pos_rank", "0").split()
        self.delete()
