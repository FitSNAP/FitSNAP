"""Basic Class"""
from ..sections import Section


class Basic(Section):

    def __init__(self, name, config, args):
        # let parent hold config and args
        super().__init__(name, config, args)
        self.num_atoms = self.get_value("BASIC", "numAtoms", "1", "int")
        self.delete()
