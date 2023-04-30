"""Custom Class"""
from fitsnap3lib.io.sections.sections import Section


class Custom(Section):

    def __init__(self, name, config, pt, infile, args):
        # let parent hold config and args
        super().__init__(name, config, pt, infile, args)
        self.num_atoms = self.get_value("CUSTOM", "numAtoms", "1", "int")
        self.numtypes = self.get_value("CUSTOM", "numTypes", "1", "int")
        self.types = self.get_value("CUSTOM", "type", "H").split()
        self.type_mapping = {}
        #print(self.types)
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1
        #print(self.type_mapping)
        self.num_descriptors = self.get_value("CUSTOM", "num_descriptors", "3", "int")
        self.num_radial = self.get_value("CUSTOM", "num_radial", "4", "int")
        self.num_3body = self.get_value("CUSTOM", "num_3body", "11", "int")
        self.num_descriptors = self.num_radial + self.num_3body
        self.cutoff = self.get_value("CUSTOM", "cutoff", "3.0", "float")

        self.delete()
