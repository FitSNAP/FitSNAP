from fitsnap3.calculators.calculator import Calculator
"""Methods you may or must override in new calculators"""


class Template(Calculator):

    def __init__(self, name):
        super().__init__(name)

    # Calculator must override process_configs method
    def process_configs(self, data, i):
        """"""
        pass
