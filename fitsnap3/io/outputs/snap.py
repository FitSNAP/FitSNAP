from .outputs import Output, optional_open
from ...parallel_tools import pt
from datetime import datetime
from ..input import config
import numpy as np


class Snap(Output):

    def __init__(self, name):
        super().__init__(name)

    def output(self, coeffs, errors):
        new_coeffs = None
        # new_coeffs = pt.combine_coeffs(coeffs)
        if new_coeffs is not None:
            coeffs = new_coeffs
        self.write(coeffs, errors)

    @pt.rank_zero
    def write(self, coeffs, errors):
        if config.sections["EXTRAS"].only_test != 1:
            if config.sections["CALCULATOR"].calculator != "LAMMPSSNAP":
                raise TypeError("SNAP output style must be paired with LAMMPSSNAP calculator")
        with optional_open(config.sections["OUTFILE"].potential_name and
                           config.sections["OUTFILE"].potential_name + '.snapcoeff', 'wt') as file:
            file.write(_to_coeff_string(coeffs))
        with optional_open(config.sections["OUTFILE"].potential_name and
                           config.sections["OUTFILE"].potential_name + '.snapparam', 'wt') as file:
            file.write(_to_param_string())
        self.write_errors(errors)

    @pt.sub_rank_zero
    def read_fit(self):
        # TODO fix this fix reader for bzeroflag = 0
        if config.sections["CALCULATOR"].calculator != "LAMMPSSNAP":
            raise TypeError("SNAP output style must be paired with LAMMPSSNAP calculator")
        with optional_open(config.sections["OUTFILE"].potential_name and
                           config.sections["OUTFILE"].potential_name + '.snapcoeff', 'r') as file:

            info = file.readline()
            toss = file.readline()
            num_types, ncoeff = [int(i) for i in file.readline().split()]
            try:
                assert ncoeff == (config.sections["BISPECTRUM"].ncoeff+1)
            except AssertionError:
                raise ValueError("number of coefficients: {} does not match "
                                 "input file ncoeff: {}".format(ncoeff, config.sections["BISPECTRUM"].ncoeff+1))
            try:
                assert num_types == config.sections["BISPECTRUM"].numtypes
            except AssertionError:
                raise ValueError("number of types: {} does not match "
                                 "input file numTypes: {}".format(num_types, config.sections["BISPECTRUM"].numtypes))
            fit = np.zeros((num_types, ncoeff-1))
            for i in range(num_types):
                atom_header = file.readline()
                zero = file.readline()
                for j in range(ncoeff-1):
                    fit[i][j] = float(file.readline().split()[0])

        return fit.flatten()


def _to_param_string():
    if config.sections["BISPECTRUM"].chemflag != 0:
        chemflag_int = 1
    else:
        chemflag_int = 0
    return f"""
    # required
    rcutfac {config.sections["BISPECTRUM"].rcutfac}
    twojmax {max(config.sections["BISPECTRUM"].twojmax)}

    #  optional
    rfac0 {config.sections["BISPECTRUM"].rfac0}
    rmin0 {config.sections["BISPECTRUM"].rmin0}
    bzeroflag {config.sections["BISPECTRUM"].bzeroflag}
    quadraticflag {config.sections["BISPECTRUM"].quadraticflag}
    wselfallflag {config.sections["BISPECTRUM"].wselfallflag}
    chemflag {chemflag_int}
    bnormflag {config.sections["BISPECTRUM"].bnormflag}
    """


def _to_coeff_string(coeffs):
    """
    Convert a set of coefficients along with bispec options into a .snapparam file
    """
    numtypes = config.sections["BISPECTRUM"].numtypes
    ncoeff = config.sections["BISPECTRUM"].ncoeff
    coeffs = coeffs.reshape((numtypes, -1))
    blank2Js = config.sections["BISPECTRUM"].blank2J.reshape((numtypes, -1))
    if config.sections["BISPECTRUM"].bzeroflag:
        blank2Js = np.insert(blank2Js, 0, [1.0], axis=1)
    coeffs = np.multiply(coeffs, blank2Js)
    type_names = config.sections["BISPECTRUM"].types
    out = f"# fitsnap fit generated on {datetime.now()}\n\n"
    out += "{} {}\n".format(len(type_names), ncoeff+1)

    for elname, rjval, wjval, column, ielem in zip(type_names,
                                            config.sections["BISPECTRUM"].radelem,
                                            config.sections["BISPECTRUM"].wj,
                                                    coeffs, range(numtypes)):
        bstart = ielem * ncoeff
        bstop = bstart + ncoeff
        bnames = [[0]] + config.sections["BISPECTRUM"].blist[bstart:bstop]

        out += "{} {} {}\n".format(elname, rjval, wjval)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, bnames))
        out += "\n"
    out += "\n# End of potential"
    return out
