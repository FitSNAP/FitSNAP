"""Custom Class"""
from fitsnap3lib.io.sections.sections import Section


class Custom(Section):

    def __init__(self, name, config, args):
        # let parent hold config and args
        super().__init__(name, config, args)
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
        # Number of descriptors due to one hot encoding is numtypes*2 since we concatenate for each ij pair.
        self.num_onehot = self.numtypes*2
        self.num_descriptors = self.num_radial + self.num_3body + self.num_onehot
        self.cutoff = self.get_value("CUSTOM", "cutoff", "3.0", "float")

        self.delete()
