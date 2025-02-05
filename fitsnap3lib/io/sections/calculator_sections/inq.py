"""Basic Class"""
from fitsnap3lib.io.sections.sections import Section


class INQ(Section):

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['theory', 'cell', 'spin', 'temperature', 'tolerance' ]
        self.theory = self.get_value("INQ", "theory", "PBE", "str")
        self.cell = self.get_value("INQ", "cell", "cubic 5 A periodic", "str")
        self.delete()
