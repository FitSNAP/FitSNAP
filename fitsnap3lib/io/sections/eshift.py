from fitsnap3lib.io.sections.sections import Section
import numpy as np
from fitsnap3lib.parallel_tools import ParallelTools


pt = ParallelTools()


class Eshift(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        types = []
        if config.has_section("BISPECTRUM"):
            self.types = self.get_value("BISPECTRUM", "type", "H").split()
        elif config.has_section("ACE"):
            self.types = self.get_value("ACE", "type", "H").split()
        elif config.has_section("CUSTOM"):
            self.types = self.get_value("CUSTOM", "type", "H").split()
            
        if config.has_section("ESHIFT"):
            self.eshift = {}
        else:
            return
        for atom_type in self.types:
            self.eshift[atom_type] = self.get_value("ESHIFT", "{}".format(atom_type), "0.0", "float")

        self.delete()
