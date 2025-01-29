"""Basic Class"""
from fitsnap3lib.io.sections.sections import Section


class Reaxff(Section):

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['potential', 'parameters']
        self.potential = self.get_value("REAXFF", "potential", "None", "str")
        self.parameters = self.get_value("REAXFF", "parameters", "None", "str")
        self.delete()
