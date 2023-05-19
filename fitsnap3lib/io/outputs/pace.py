from fitsnap3lib.io.outputs.outputs import Output, optional_open
from datetime import datetime
import numpy as np
import itertools

#config = Config()
#pt = ParallelTools()

try:

    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot

    class Pace(Output):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            #self.config = Config()
            #self.pt = ParallelTools()

        def output(self, coeffs, errors):
            if (self.config.sections["CALCULATOR"].nonlinear):
                self.write_nn(errors)
            else:
                new_coeffs = None
                # new_coeffs = pt.combine_coeffs(coeffs)
                if new_coeffs is not None:
                    coeffs = new_coeffs
                self.write(coeffs, errors)

        def write_lammps(self, coeffs):
            """
            Write LAMMPS ready ACE files.

            Args:
                coeffs: list of linear model coefficients.
            """
            if self.config.sections["EXTRAS"].only_test != 1:
                if self.config.sections["CALCULATOR"].calculator != "LAMMPSPACE":
                    raise TypeError("PACE output style must be paired with LAMMPSPACE calculator")
                with optional_open(self.config.sections["OUTFILE"].potential_name and
                                  self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                    file.write(_to_coeff_string(coeffs, self.config))
                self.write_potential(coeffs)

        #@pt.rank_zero
        def write(self, coeffs, errors):
            @self.pt.rank_zero
            def decorated_write():
                """
                if self.config.sections["EXTRAS"].only_test != 1:
                    if self.config.sections["CALCULATOR"].calculator != "LAMMPSPACE":
                        raise TypeError("PACE output style must be paired with LAMMPSPACE calculator")
                    with optional_open(self.config.sections["OUTFILE"].potential_name and
                                      self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                        file.write(_to_coeff_string(coeffs, self.config))
                    self.write_potential(coeffs)
                """
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
                # TODO: Add mliap decriptor writing when LAMMPS implementation of NN-ACE is complete.
                self.write_errors_nn(errors)
            decorated_write()

        #@pt.sub_rank_zero
        def read_fit(self):
            @self.pt.sub_rank_zero
            def decorated_read_fit():
                assert NotImplementedError("read fit for pace potentials not implemented")
            decorated_read_fit()


        def write_potential(self, coeffs):

            self.bzeroflag = self.config.sections["ACE"].bzeroflag 
            self.numtypes = self.config.sections["ACE"].numtypes
            self.ranks = self.config.sections["ACE"].ranks
            self.lmin = self.config.sections["ACE"].lmin
            self.lmax = self.config.sections["ACE"].lmax
            self.nmax = self.config.sections["ACE"].nmax
            self.mumax = self.config.sections["ACE"].mumax
            self.nmaxbase = self.config.sections["ACE"].nmaxbase
            self.rcutfac = self.config.sections["ACE"].rcutfac
            self.lmbda =self.config.sections["ACE"].lmbda
            self.rcinner = self.config.sections["ACE"].rcinner
            self.drcinner = self.config.sections["ACE"].drcinner
            self.types = self.config.sections["ACE"].types
            self.erefs = self.config.sections["ACE"].erefs
            self.bikflag = self.config.sections["ACE"].bikflag
            self.RPI_heuristic = self.config.sections["ACE"].RPI_heuristic

            if self.bzeroflag:
                assert len(self.types) ==  len(self.erefs), "must provide reference energy for each atom type"
                reference_ens = [float(e0) for e0 in self.erefs]
            elif not self.bzeroflag:
                reference_ens = [0.0] * len(self.types)
            bondinds=range(len(self.types))
            bonds = [b for b in itertools.product(bondinds,bondinds)]
            bondstrs = ['[%d, %d]' % b for b in bonds]
            assert len(self.lmbda) == len(bondstrs), "must provide rc, lambda, for each BOND type"
            assert len(self.rcutfac) == len(bondstrs), "must provide rc, lambda, for each BOND type"
            if len(self.lmbda) == 1:
                lmbdavals = self.lmbda
                rcvals = self.rcutfac
                rcinnervals = self.rcinner
                drcinnervals = self.drcinner
            if len(self.lmbda) > 1:
                lmbdavals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.lmbda)}
                rcvals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcutfac)}
                rcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcinner)}
                drcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.drcinner)}

            apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, [int(k) for k in self.lmin], self.RPI_heuristic)
            apot.set_betas(coeffs,has_zeros=True)
            apot.set_funcs()
            apot.write_pot(self.config.sections["OUTFILE"].potential_name)
            # Append metadata to .yace file
            unit = f"# units {self.config.sections['REFERENCE'].units}\n"
            atom = f"# atom_style {self.config.sections['REFERENCE'].atom_style}\n"
            pair = "\n".join(["# " + s for s in self.config.sections["REFERENCE"].lmp_pairdecl]) + "\n"
            refsec = unit + atom + pair
            with open(f"{self.config.sections['OUTFILE'].potential_name}.yace", "a") as fp:
                fp.write("# This file was generated by FitSNAP.\n")
                fp.write(f"# Hash: {self.config.hash}\n")
                fp.write(f"# FitSNAP REFERENCE section settings:\n")
                fp.write(f"{refsec}")

except ModuleNotFoundError:

    class Pace(Output):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("Missing sympy or pyyaml modules.")

def _to_coeff_string(coeffs, config):
    """
    Convert a set of coefficients along with descriptor options to a coeffs file.
    """

    desc_str = "ACE"
    coeffs = coeffs.reshape((config.sections[desc_str].numtypes, -1))
    blank2Js = config.sections[desc_str].blank2J.reshape((config.sections[desc_str].numtypes, -1))
    if config.sections[desc_str].bzeroflag:
        blank2Js = np.insert(blank2Js, 0, [1.0], axis=1)
    coeffs = np.multiply(coeffs, blank2Js)
    coeff_names = [[0]]+config.sections[desc_str].blist
    type_names = config.sections[desc_str].types
    out = f"# FitSNAP generated on {datetime.now()} with Hash: {config.hash}\n\n"
    out += "{} {}\n".format(len(type_names), int(np.ceil(len(coeff_names)/config.sections[desc_str].numtypes)))
    for elname, column in zip(type_names,
                        coeffs):
        out += "{}\n".format(elname)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, coeff_names))
        out += "\n"
    out += "\n# End of potential"
    return out
