from fitsnap3.io.sections.sections import Section


class Reference(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.units = self._config.get("REFERENCE", "units", fallback="metal").lower()
        self.atom_style = self._config.get("REFERENCE", "atom_style", fallback="atomic").lower()
        self.lmp_pairdecl = []
        self.lmp_pairdecl.append("pair_style " + self._config.get("REFERENCE", "pair_style", fallback="zero 10.0"))
        for name, value in self._config.items("REFERENCE"):
            if not name.find("pair_coeff"):
                self.lmp_pairdecl.append("pair_coeff " + value)
        if "pair_coeff" in self.lmp_pairdecl:
            self.lmp_pairdecl.append("pair_coeff * * ")
        self.delete()
