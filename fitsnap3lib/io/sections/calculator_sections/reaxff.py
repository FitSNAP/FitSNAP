"""Basic Class"""
from fitsnap3lib.io.sections.sections import Section


class Reaxff(Section):

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)

        self.allowedkeys = ['force_field', 'parameters']
        self.force_field = self.get_value("REAXFF", "force_field", "None", "str")
        self.parameters = self.get_value("REAXFF", "parameters", "None", "str")
        self.numtypes = 3
        self.type_mapping = {}
        self.type_mapping['H'] = 1
        self.type_mapping['O'] = 2
        self.type_mapping['X'] = 3

        self.delete()
