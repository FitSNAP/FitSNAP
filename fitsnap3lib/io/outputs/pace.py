from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.parallel_tools import ParallelTools
from datetime import datetime
from fitsnap3lib.io.input import Config
import numpy as np


#config = Config()
#pt = ParallelTools()


class Pace(Output):

    def __init__(self, name):
        super().__init__(name)

    def output(self, coeffs, errors):
        new_coeffs = None
        # new_coeffs = pt.combine_coeffs(coeffs)
        if new_coeffs is not None:
            coeffs = new_coeffs
        self.write(coeffs, errors)

    #@pt.rank_zero
    def write(self, coeffs, errors):
        if self.config.sections["EXTRAS"].only_test != 1:
            if self.config.sections["CALCULATOR"].calculator != "LAMMPSPACE":
                raise TypeError("PACE output style must be paired with LAMMPSPACE calculator")
            with optional_open(self.config.sections["OUTFILE"].potential_name and
                               self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                file.write(_to_coeff_string(coeffs))
        self.write_errors(errors)

    #@pt.sub_rank_zero
    def read_fit(self):
        assert NotImplementedError("read fit for pace potentials not implemented")


def _to_coeff_string(coeffs):
    """
    Convert a set of coefficients along with bispec options into a .snapparam file
    """
    desc_str = "ACE"

    coeffs = coeffs.reshape((self.config.sections[desc_str].numtypes, -1))
    blank2Js = self.config.sections[desc_str].blank2J.reshape((self.config.sections[desc_str].numtypes, -1))
    if self.config.sections[desc_str].bzeroflag:
        blank2Js = np.insert(blank2Js, 0, [1.0], axis=1)
    coeffs = np.multiply(coeffs, blank2Js)
    coeff_names = [[0]]+self.config.sections[desc_str].blist
    type_names = self.config.sections[desc_str].types
    out = f"# fitsnap fit/pace generated on {datetime.now()}\n\n"
    out += "{} {}\n".format(len(type_names), int(np.ceil(len(coeff_names)/self.config.sections[desc_str].numtypes)))
    for elname, column in zip(type_names,
                        coeffs):
        out += "{}\n".format(elname)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, coeff_names))
        out += "\n"
    out += "\n# End of potential"
    return out
