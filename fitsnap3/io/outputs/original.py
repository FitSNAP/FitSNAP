from fitsnap3.io.outputs.outputs import Output, optional_write
from fitsnap3.parallel_tools import pt
from datetime import datetime
from fitsnap3.io.input import config


class Original(Output):

    def __init__(self, name):
        super().__init__(name)

    def output(self, coeffs, errors):
        new_coeffs = pt.combine_coeffs(coeffs)
        if new_coeffs is not None:
            coeffs = new_coeffs
        self.write(coeffs, errors)

    @pt.rank_zero
    def write(self, coeffs, errors):
        with optional_write(config.sections["OUTFILE"].potential_name and
                            config.sections["OUTFILE"].potential_name + '.snapcoeff', 'wt') as file:
            file.write(_to_coeff_string(coeffs))
        with optional_write(config.sections["OUTFILE"].potential_name and
                            config.sections["OUTFILE"].potential_name + '.snapparam', 'wt') as file:
            file.write(_to_param_string())
        with optional_write(config.sections["OUTFILE"].metric_file, 'wt') as file:
            errors.to_csv(file)


def _to_param_string():
    if config.sections["CALCULATOR"].chemflag != 0:
        chemflag_int = 1
    else:
        chemflag_int = 0
    return f"""
    # required
    rcutfac {config.sections["BISPECTRUM"].rcutfac}
    twojmax {config.sections["BISPECTRUM"].twojmax}

    #  optional
    rfac0 {config.sections["BISPECTRUM"].rfac0}
    rmin0 {config.sections["BISPECTRUM"].rmin0}
    bzeroflag {config.sections["CALCULATOR"].bzeroflag}
    quadraticflag {config.sections["CALCULATOR"].quadraticflag}
    wselfallflag {config.sections["CALCULATOR"].wselfallflag}
    chemflag {chemflag_int}
    bnormflag {config.sections["CALCULATOR"].bnormflag}
    """


def _to_coeff_string(coeffs):
    """
    Convert a set of coefficients along with bispec options into a .snapparam file
    """
    # Includes the offset term, which was not in blist
    coeffs = coeffs.reshape((config.sections["BISPECTRUM"].numtypes, -1))
    coeff_names = [[0]]+config.sections["BISPECTRUM"].blist
    type_names = config.sections["BISPECTRUM"].types
    out = f"# fitsnap fit generated on {datetime.now()}\n\n"
    out += "{} {}\n".format(len(type_names), len(coeff_names))
    for elname, rjval, wjval, column in zip(type_names,
                                            config.sections["BISPECTRUM"].radelem,
                                            config.sections["BISPECTRUM"].wj,
                                            coeffs):
        out += "{} {} {}\n".format(elname, rjval, wjval)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, coeff_names))
        out += "\n"
    out += "\n# End of potential"
    return out
