from fitsnap3.io.sections.sections import Section


class Eshift(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        if "ESHIFT" in self._config:
            self._set_eshifts()
        self.delete()

    def _set_eshifts(self):
        types = []
        for i in range(int(self._config.get("BISPECTRUM", "numTypes", fallback='1'))):
            types.append(self._config.get("BISPECTRUM", "type{}".format(i + 1), fallback='H'))
        self.eshift = {}
        for atom_type in types:
            self.eshift[atom_type] = float(self._config.get("ESHIFT", "{}".format(atom_type), fallback='0'))
