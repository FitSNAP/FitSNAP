from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
import numpy as np


config = Config()
pt = ParallelTools()


class Basic(Calculator):

    def __init__(self, name):
        super().__init__(name)
        self._data = {}
        self._array = None
        self._i = 0

    def create_a(self):
        super().create_a()

    # Calculator must override process_configs method
    def process_configs(self, data, i):
        self._data = data
        self._array = np.zeros((config.sections["BASIC"].num_atoms*3-6))
        self._i = i
        self.convert_to_internal()
        self.add_to_a()

    def convert_to_internal(self):
        a, b, c = np.zeros(3), np.zeros(3), np.zeros(3)
        for i, (a_x, a_y, a_z) in enumerate(self._data["Positions"]):
            if i == 0:
                c[:] = a_x, a_y, a_z
            elif i == 1:
                b[:] = a_x, a_y, a_z
                self._array[0] = np.linalg.norm(b - c)
            elif i == 2:
                a[:] = a_x, a_y, a_z
                self._array[1] = np.linalg.norm(a - c)
                self._array[2] = np.arccos(((a-c)@(b-c))/(np.linalg.norm(a-c)*np.linalg.norm(b-c)))
            else:
                this = np.array([a_x, a_y, a_z])
                self._array[i*3-6] = np.linalg.norm(this - c)
                self._array[i*3-5] = np.arccos(((this - c) @ (b - c)) / (np.linalg.norm(this - c) * np.linalg.norm(b - c)))
                self._array[i*3-4] = np.arccos(((this - b) @ (a - b)) / (np.linalg.norm(this - b) * np.linalg.norm(a - b)))
                c = b
                b = a
                a = this

    def add_to_a(self):
        if config.sections["CALCULATOR"].energy:
            pt.shared_arrays['a'].array[self._i] = self._array
            pt.shared_arrays['b'].array[self._i] = self._data["Energy"]
            pt.shared_arrays['w'].array[self._i] = self._data["eweight"]

    def get_width(self):
        return config.sections["BASIC"].num_atoms*3-6
