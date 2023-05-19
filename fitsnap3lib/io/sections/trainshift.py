from fitsnap3lib.io.sections.sections import Section
import numpy as np
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Trainshift(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)

        if config.has_section("BISPECTRUM"):
            self.types = self.get_value("BISPECTRUM", "type", "H").split()
        elif config.has_section("ACE"):
            self.types = self.get_value("ACE", "type", "H").split()
        elif config.has_section("CUSTOM"):
            self.types = self.get_value("CUSTOM", "type", "H").split()

        if config.has_section("TRAINSHIFT"):
            self.trainshift = {}
        else:
            return

        ## Hacky way to get multiple data types into this section
        for atom_type in self.types: 
            self.trainshift[atom_type] = self.get_value("TRAINSHIFT", "{}".format(atom_type), "0.0", "float")

        self.delete()
