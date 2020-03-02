from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config
import numpy as np


class Solver:

    def __init__(self, name):
        self.name = name
        self.fit = None
        self.configs = None

    def perform_fit(self):
        pass

    def _offset(self):
        num_types = config.sections["BISPECTRUM"].numtypes
        if num_types > 1:
            self.fit = self.fit.reshape(num_types, config.sections["BISPECTRUM"].ncoeff)
            offsets = np.zeros((num_types, 1))
            self.fit = np.concatenate([offsets, self.fit], axis=1)
            self.fit = self.fit.reshape((-1, 1))
        else:
            self.fit = np.insert(self.fit, 0, 0)

    def error_analysis(self, data):
        pass
