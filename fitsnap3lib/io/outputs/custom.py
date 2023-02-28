from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.parallel_tools import ParallelTools
from datetime import datetime
from fitsnap3lib.io.input import Config
import numpy as np
import random
import tarfile


#config = Config()
#pt = ParallelTools()


class Custom(Output):

    def __init__(self, name):
        super().__init__(name)
        self.config = Config()
        self.pt = ParallelTools()

    def output(self, coeffs, errors):
        self.write_nn(errors)

    #@pt.rank_zero
    def write(self, coeffs, errors):
        @self.pt.rank_zero
        def decorated_write():
            if self.config.sections["EXTRAS"].only_test != 1:
                if self.config.sections["CALCULATOR"].calculator != "LAMMPSCUSTOM":
                    raise TypeError("CUSTOM output style must be paired with LAMMPSCUSTOM calculator")
            self.write_errors(errors)
        decorated_write()

    def write_nn(self, errors):
        """ 
        Write output for nonlinear fits. 
        
        Args:
            errors : sequence of dictionaries (group_mae_f, group_mae_e, group_rmse_e, group_rmse_f)
        """
        @self.pt.rank_zero
        def decorated_write():
            if self.config.sections["EXTRAS"].only_test != 1:
                if (self.config.sections["CALCULATOR"].calculator != "LAMMPSCUSTOM"):
                    raise TypeError("CUSTOM output style must be paired with LAMMPSCUSTOM calculator")
            self.write_errors_nn(errors)
        decorated_write()

    #@pt.sub_rank_zero
    def read_fit(self):
        @self.pt.sub_rank_zero
        def decorated_read_fit():
            # TODO: Not implemented for custom networks
            pass