from ..io.outputs.output_factory import output_factory
from ..io.input import Config


config = Config()


if __name__.split(".")[-1] == "output":
    output = output_factory(config.sections["OUTFILE"].output_style)
