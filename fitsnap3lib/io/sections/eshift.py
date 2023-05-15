from fitsnap3lib.io.sections.sections import Section
import numpy as np


class Eshift(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
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
