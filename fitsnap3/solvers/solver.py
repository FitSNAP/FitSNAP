from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config


class Solver:

    def __init__(self, name):
        self.name = name
        self.fit = None
        self.configs = None

    def perform_fit(self):
        pass

    def error_analysis(self, data):
        pass
