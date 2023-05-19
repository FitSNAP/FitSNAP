from fitsnap3lib.io.sections.sections import Section
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Reference(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['units', 'atom_style', 'pair_style', 'pair_coeff']

        # for value_name in config['REFERENCE']:
        #     if value_name in allowedkeys: continue
        #     else: pt.single_print(">>> Found unmatched variable in REFERENCE section of input: ",value_name)

        self.units = self.get_value("REFERENCE", "units", "metal").lower()
        self.atom_style = self.get_value("REFERENCE", "atom_style", "atomic").lower()
        self.lmp_pairdecl = []
        self.lmp_pairdecl.append("pair_style " + self.get_value("REFERENCE", "pair_style", "zero 10.0"))
        if not config.has_section("REFERENCE"):
            self.delete()
            return
        for name, value in self._config.items("REFERENCE"):
            if not name.find("pair_coeff"):
                self.lmp_pairdecl.append("pair_coeff " + value)
        if "pair_coeff" in self.lmp_pairdecl:
            self.lmp_pairdecl.append("pair_coeff * * ")
        self.delete()
