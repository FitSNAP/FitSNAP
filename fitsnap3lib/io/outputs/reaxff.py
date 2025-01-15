from fitsnap3lib.io.outputs.outputs import Output, optional_open
"""Methods you may or must override in new output"""


class Reaxff(Output):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    # You must override this method
    def output(self, coeffs, errors):

        #print( errors )

        @self.pt.rank_zero
        def write_ff():
            with optional_open(self.config.sections["OUTFILE"].potential_name, 'wt') as file:
                file.write(coeffs)

        write_ff()


