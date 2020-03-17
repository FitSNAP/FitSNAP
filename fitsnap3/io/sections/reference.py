from fitsnap3.io.sections.sections import Section


class Reference(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.units = self.get_value("REFERENCE", "units", "metal").lower()
        self.atom_style = self.get_value("REFERENCE", "atom_style", "atomic").lower()
        self.lmp_pairdecl = []
        self.lmp_pairdecl.append("pair_style " + self.get_value("REFERENCE", "pair_style", "zero 10.0"))
        for name, value in self._config.items("REFERENCE"):
            if not name.find("pair_coeff"):
                self.lmp_pairdecl.append("pair_coeff " + value)
        if "pair_coeff" in self.lmp_pairdecl:
            self.lmp_pairdecl.append("pair_coeff * * ")
        self.delete()
