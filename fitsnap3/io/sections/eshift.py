from .sections import Section
import numpy as np
from ...parallel_tools import pt

class Eshift(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        types = []
        #for i in range(self.get_value("BISPECTRUM", "numTypes", "1", "int")):
        #    types.append(self.get_value("BISPECTRUM", "type{}".format(i + 1), "H"))
        self.types = self.get_value("BISPECTRUM", "type", "H").split()
        #self.types = np.reshape(self.types,(self.get_value("BISPECTRUM", "numTypes", "1", "int"),1))
        if config.has_section("ESHIFT"):
            self.eshift = {}
        else:
            return
        for atom_type in self.types:
            self.eshift[atom_type] = self.get_value("ESHIFT", "{}".format(atom_type), "0.0", "float")

        self.delete()
