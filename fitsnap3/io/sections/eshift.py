from fitsnap3.io.sections.sections import Section


class Eshift(Section):

    def __init__(self, name, config, args):
        self._on = "False"
        super().__init__(name, config, args)
        types = []
        for i in range(self.get_value("BISPECTRUM", "numTypes", "1", "int")):
            types.append(self.get_value("BISPECTRUM", "type{}".format(i + 1), "H"))
        self.eshift = {}
        for atom_type in types:
            self.eshift[atom_type] = self.get_value("ESHIFT", "{}".format(atom_type), "0.0", "float")
        self.delete()
