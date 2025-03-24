from fitsnap3lib.io.outputs.outputs import Output, optional_open
import os


class Reaxff(Output):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    # You must override this method
    def output(self, coeffs, errors):

        #print( errors )

        @self.pt.rank_zero
        def write_ff():
            base, ext = os.path.splitext(self.config.sections["OUTFILE"].potential_name)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            with optional_open(f"{base}-{timestamp}{ext}", 'wt') as file:
                file.write(coeffs)

        write_ff()


