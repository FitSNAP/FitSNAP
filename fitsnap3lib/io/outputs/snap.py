from fitsnap3lib.io.outputs.outputs import Output, optional_open
from datetime import datetime
import numpy as np
import random
import tarfile


class Snap(Output):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    def output(self, coeffs, errors):
        if (self.config.sections["CALCULATOR"].nonlinear):
            # Currently coeffs and errors are empty for nonlinear.
            # TODO: Add nonlinear error calculation for output here, similar format as linear.
            self.write_nn(errors)
        else:
            new_coeffs = None
            # new_coeffs = pt.combine_coeffs(coeffs)
            if new_coeffs is not None:
                coeffs = new_coeffs
            self.write(coeffs, errors)

    def write_lammps(self, coeffs):
        """
        Write LAMMPS-ready SNAP files.
        
        Args:
            coeffs: list of linear model coefficients.
        """
        if self.config.sections["EXTRAS"].only_test != 1:
            if self.config.sections["CALCULATOR"].calculator != "LAMMPSSNAP":
                raise TypeError("SNAP output style must be paired with LAMMPSSNAP calculator")
        with optional_open(self.config.sections["OUTFILE"].potential_name and
                            self.config.sections["OUTFILE"].potential_name + '.snapcoeff', 'wt') as file:
            file.write(_to_coeff_string(self.config, coeffs))
        with optional_open(self.config.sections["OUTFILE"].potential_name and
                            self.config.sections["OUTFILE"].potential_name + '.snapparam', 'wt') as file:
            file.write(_to_param_string(self.config))
        with optional_open(self.config.sections["OUTFILE"].potential_name and
                            self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
            file.write(_to_potential_file(self.config))
        if (self._tarball):
            with optional_open("in.lammps", 'wt') as file:
                file.write(_to_lammps_input())
            # Package these files into a tarball
            fp = tarfile.open(f"fit-{self.config.hash}.tar.gz", 'w:gz')
            potname = self.config.sections["OUTFILE"].potential_name
            potname_prefix = potname.split('/')[-1]
            fp.add(potname + '.snapcoeff', arcname = potname_prefix + '.snapcoeff')
            fp.add(potname + '.snapparam', arcname = potname_prefix + '.snapparam')
            fp.add(potname + '.mod', arcname = potname_prefix)
            fp.add("in.lammps")
            fp.close()

    #@pt.rank_zero
    def write(self, coeffs, errors):
        """ Write both LAMMPS files and error files"""
        @self.pt.rank_zero
        def decorated_write():
            self.write_lammps(coeffs)
            self.write_errors(errors)
        decorated_write()

    def write_nn(self, errors):
        """ 
        Write output for nonlinear fits. 
        
        Args:
            errors : sequence of dictionaries (group_mae_f, group_mae_e, group_rmse_e, group_rmse_f)
        """
        @self.pt.rank_zero
        def decorated_write():
            if self.config.sections["EXTRAS"].only_test != 1:
                if (self.config.sections["CALCULATOR"].calculator != "LAMMPSSNAP"):
                    raise TypeError("SNAP output style must be paired with LAMMPSSNAP calculator")
            with optional_open(self.config.sections["OUTFILE"].potential_name and
                               self.config.sections["OUTFILE"].potential_name + '.mliap.descriptor', 'wt') as file:
                file.write(_to_mliap_string(self.config))
            with optional_open(self.config.sections["OUTFILE"].potential_name and
                               self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
                file.write(_to_mliap_mod(self.config))

            self.write_errors_nn(errors)
        decorated_write()

    #@pt.sub_rank_zero
    def read_fit(self):
        @self.pt.sub_rank_zero
        def decorated_read_fit():
            # TODO fix this fix reader for bzeroflag = 0
            if self.config.sections["CALCULATOR"].calculator != "LAMMPSSNAP":
                raise TypeError("SNAP output style must be paired with LAMMPSSNAP calculator")
            with optional_open(self.config.sections["OUTFILE"].potential_name and
                               self.config.sections["OUTFILE"].potential_name + '.snapcoeff', 'r') as file:

                info = file.readline()
                toss = file.readline()
                num_types, ncoeff = [int(i) for i in file.readline().split()]
                try:
                    assert ncoeff == (self.config.sections["BISPECTRUM"].ncoeff+1)
                except AssertionError:
                    raise ValueError("number of coefficients: {} does not match "
                                     "input file ncoeff: {}".format(ncoeff, self.config.sections["BISPECTRUM"].ncoeff+1))
                try:
                    assert num_types == self.config.sections["BISPECTRUM"].numtypes
                except AssertionError:
                    raise ValueError("number of types: {} does not match "
                                     "input file numTypes: {}".format(num_types, self.config.sections["BISPECTRUM"].numtypes))
                fit = np.zeros((num_types, ncoeff-1))
                for i in range(num_types):
                    atom_header = file.readline()
                    zero = file.readline()
                    for j in range(ncoeff-1):
                        fit[i][j] = float(file.readline().split()[0])
            return fit
        fit = decorated_read_fit()
        return fit.flatten()


def _to_param_string(config):
    if config.sections["BISPECTRUM"].chemflag != 0:
        chemflag_int = 1
    else:
        chemflag_int = 0
    # Report reference section settings
    unit = f"# units {config.sections['REFERENCE'].units}\n"
    atom = f"# atom_style {config.sections['REFERENCE'].atom_style}\n"
    pair = "\n".join(["# " + s for s in config.sections["REFERENCE"].lmp_pairdecl]) + "\n"
    refsec = unit + atom + pair
    # Build output for param file
    out = "# required\n"
    out += f"rcutfac {config.sections['BISPECTRUM'].rcutfac}\n"
    out += f"twojmax {max(config.sections['BISPECTRUM'].twojmax)}\n\n"
    out += "# optional\n"
    out += f"rfac0 {config.sections['BISPECTRUM'].rfac0}\n"
    out += f"rmin0 {config.sections['BISPECTRUM'].rmin0}\n"
    out += f"bzeroflag {config.sections['BISPECTRUM'].bzeroflag}\n"
    out += f"wselfallflag {config.sections['BISPECTRUM'].wselfallflag}\n"
    out += f"chemflag {chemflag_int}\n"
    out += f"bnormflag {config.sections['BISPECTRUM'].bnormflag}\n"
    out += f"switchinnerflag {config.sections['BISPECTRUM'].switchinnerflag}\n"
    out += f"quadraticflag {config.sections['BISPECTRUM'].quadraticflag}\n"
    if config.sections["BISPECTRUM"].switchinnerflag:
        out += f"sinner {config.sections['BISPECTRUM'].sinner}\n"
        out += f"dinner {config.sections['BISPECTRUM'].dinner}\n"
    out += "\n"
    out += "# This file was generated by FitSNAP.\n"
    out += f"# Hash: {config.hash}\n"
    out += "# FitSNAP REFERENCE section settings:\n"
    out += f"{refsec}"

    return out

def _to_coeff_string(config, coeffs):
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
    out = f"# FitSNAP fit generated on {datetime.now()} with Hash: {config.hash}\n\n"
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

def _to_potential_file(config):
    """
    Use config settings to write a LAMMPS potential .mod file.
    """

    ps = config.sections["REFERENCE"].lmp_pairdecl[0]
    snap_filename = config.sections["OUTFILE"].potential_name.split("/")[-1]

    out = "# This file was generated by FitSNAP.\n"
    out += f"# Hash: {config.hash}\n\n"

    if "hybrid" in ps:
        # extract non zero parts of pair style
        if "zero" in ps.split():
            split = ps.split()
            zero_indx = split.index("zero")
            del split[zero_indx]
            del split[zero_indx] # delete the zero pair cutoff
            ps = ' '.join(split)
        out += ps + " snap\n"
        # add pair coeff commands from input, ignore if pair zero
        for pc in config.sections["REFERENCE"].lmp_pairdecl[1:]:
            out += f"{pc}\n" if "zero" not in pc else ""
        pc_snap = f"pair_coeff * * snap {snap_filename}.snapcoeff {snap_filename}.snapparam"
        for t in config.sections["BISPECTRUM"].types:
            pc_snap += f" {t}"
        out += pc_snap
    else:
        out += "pair_style snap\n"
        pc_snap = f"pair_coeff * * {snap_filename}.snapcoeff {snap_filename}.snapparam"
        for t in config.sections["BISPECTRUM"].types:
            pc_snap += f" {t}"
        out += pc_snap

    return out

def _to_lammps_input(config):
    """
    Use config settings to write a LAMMPS input script.
    """

    snap_filename = config.sections["OUTFILE"].potential_name.split("/")[-1]
    pot_filename = snap_filename + ".mod"

    out = "# LAMMPS template input written by FitSNAP.\n"
    out += "# Runs a NVE simulation at specified temperature and timestep.\n"
    out += "\n"
    out += "# Declare simulation variables\n"
    out += "\n"
    out += "variable timestep equal 0.5e-3\n"
    out += "variable temperature equal 600\n"
    out += "\n"
    out += f"units {config.sections['REFERENCE'].units}\n"
    out += f"atom_style {config.sections['REFERENCE'].atom_style}\n"
    out += "\n"
    out += "# Supply your own data file below\n"
    out += "\n"
    out += "read_data DATA\n"
    out += "\n"
    out += "# Include potential file\n"
    out += "\n"
    out += f"include {pot_filename}\n"
    out += "\n"
    out += "# Declare simulation settings\n"
    out += "\n"
    out += "timestep ${timestep}\n"
    out += "neighbor 1.0 bin\n"
    out += "velocity all create ${temperature} 10101 rot yes mom yes\n"
    out += "fix 1 all nve\n"
    out += "\n"
    out += "# Run dynamics\n"
    out += "\n"
    out += "run 1000\n"

    return out

def _to_mliap_string(config):
    """ Build mliap descriptor file. """
    out = "# required\n"
    out += f"rcutfac {config.sections['BISPECTRUM'].rcutfac}\n"
    out += f"twojmax {max(config.sections['BISPECTRUM'].twojmax)}\n\n"
    out += "#elements\n"
    out += f"nelems {config.sections['BISPECTRUM'].numtypes}\n"
    out += "elems"
    for t in config.sections['BISPECTRUM'].types:
        out += f" {t}"
    out += "\n"
    out += "radelems"
    for t in range(config.sections['BISPECTRUM'].numtypes):
        out += f" {config.sections['BISPECTRUM'].radelem[t]}"
    out += "\n"
    out += "welems"
    for t in range(config.sections['BISPECTRUM'].numtypes):
        out += f" {config.sections['BISPECTRUM'].wj[t]}"
    out += "\n"
    if config.sections["BISPECTRUM"].switchinnerflag:
        out += f"sinnerelems {config.sections['BISPECTRUM'].sinner}\n"
        out += f"dinnerelems {config.sections['BISPECTRUM'].dinner}\n"
    out += "\n\n"
    out += "# optional\n"
    out += f"rfac0 {config.sections['BISPECTRUM'].rfac0}\n"
    out += f"rmin0 {config.sections['BISPECTRUM'].rmin0}\n"
    out += f"switchinnerflag {config.sections['BISPECTRUM'].switchinnerflag}\n"
    out += f"bzeroflag {config.sections['BISPECTRUM'].bzeroflag}\n\n"

    # Report reference section settings.
    unit = f"# units {config.sections['REFERENCE'].units}\n"
    atom = f"# atom_style {config.sections['REFERENCE'].atom_style}\n"
    pair = "\n".join(["# " + s for s in config.sections["REFERENCE"].lmp_pairdecl]) + "\n"
    refsec = unit + atom + pair
    out += f"# FitSNAP generated Hash: {config.hash}\n"
    out += "# FitSNAP REFERENCE section settings:\n"
    out += f"{refsec}"
    return out

def _to_mliap_mod(config):
    """ Build mliap mod file for using the potential. """
    ps = config.sections["REFERENCE"].lmp_pairdecl[0]
    snap_filename = config.sections["OUTFILE"].potential_name.split("/")[-1]

    out = f"# FitSNAP generated Hash: {config.hash}\n"

    if "hybrid" in ps:
        # extract non zero parts of pair style
        if "zero" in ps.split():
            split = ps.split()
            zero_indx = split.index("zero")
            del split[zero_indx]
            del split[zero_indx] # delete the zero pair cutoff
            ps = ' '.join(split)
        out += ps + f" mliap model mliappy FitTorch_Pytorch.pt descriptor sna {snap_filename}.mliap.descriptor\n"
        # add pair coeff commands from input, ignore if pair zero
        for pc in config.sections["REFERENCE"].lmp_pairdecl[1:]:
            out += f"{pc}\n" if "zero" not in pc else ""
        pc_snap = f"pair_coeff * * mliap"
        for t in config.sections["BISPECTRUM"].types:
            pc_snap += f" {t}"
        out += pc_snap
    else:
        out += f"pair_style mliap model mliappy FitTorch_Pytorch.pt descriptor sna {snap_filename}.mliap.descriptor\n"
        pc_snap = f"pair_coeff * *"
        for t in config.sections["BISPECTRUM"].types:
            pc_snap += f" {t}"
        out += pc_snap

    return out
