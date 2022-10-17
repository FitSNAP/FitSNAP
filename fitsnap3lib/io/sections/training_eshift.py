from fitsnap3lib.io.sections.sections import Section
import numpy as np
from fitsnap3lib.parallel_tools import ParallelTools


pt = ParallelTools()


class Training_Eshift(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        # types = []
        if config.has_section("ESHIFT"):
            self.training_eshift = {}
        else:
            return
        for atom_type in self.types:
            self.training_eshift[atom_type] = self.get_value("TRAINING_ESHIFT", "{}".format(atom_type), "0.0", "float")

        self.delete()
