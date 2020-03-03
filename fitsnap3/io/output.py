from fitsnap3.io.outputs.output_factory import output_factory
from fitsnap3.io.input import config

if __name__ == "fitsnap3.io.output":
    output = output_factory(config.sections["OUTFILE"].output_style)
