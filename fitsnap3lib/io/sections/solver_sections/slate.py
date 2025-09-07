from fitsnap3lib.io.sections.sections import Section


class Slate(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        self.allowedkeys = ['alpha']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SLATE")

        self.alpha = self.get_value("SLATE", "alpha", "1.0E-8", "float")
        self.delete()
