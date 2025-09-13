from fitsnap3lib.io.outputs.outputs import Output, optional_open
from datetime import datetime
import numpy as np
import itertools
import pickle

#config = Config()
#pt = ParallelTools()

try:

    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import AcePot
    from fitsnap3lib.lib.sym_ACE.wigner_couple import *
    from fitsnap3lib.lib.sym_ACE.clebsch_couple import *

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
                if self.config.sections["CALCULATOR"].calculator not in ["LAMMPSPACE", "PYACE"]:
                    raise TypeError("PACE output style must be paired with LAMMPSPACE or PYACE calculator")
                
                # For ACE section, write both .acecoeff and potential files
                if "ACE" in self.config.sections and hasattr(self.config.sections["ACE"], 'blank2J'):
                    with optional_open(self.config.sections["OUTFILE"].potential_name and
                                      self.config.sections["OUTFILE"].potential_name + '.acecoeff', 'wt') as file:
                        file.write(_to_coeff_string(coeffs, self.config))
                    self.write_potential(coeffs)
                    with optional_open(self.config.sections["OUTFILE"].potential_name and self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
                        file.write(_to_potential_file(self.config))
                # For PYACE section, only write potential files (skip .acecoeff)
                elif "PYACE" in self.config.sections:
                    self.pt.single_print("PYACE detected: Skipping .acecoeff output, generating .yace potential file only")
                    self.write_potential(coeffs)
                    with optional_open(self.config.sections["OUTFILE"].potential_name and self.config.sections["OUTFILE"].potential_name + '.mod', 'wt') as file:
                        file.write(_to_potential_file(self.config))
                else:
                    raise RuntimeError("No supported ACE or PYACE section found for output")

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
            
            # Support both ACE and PYACE sections
            if "ACE" in self.config.sections:
                ace_section = self.config.sections["ACE"]
            elif "PYACE" in self.config.sections:
                ace_section = self.config.sections["PYACE"]
            else:
                raise RuntimeError("No ACE or PYACE section found for PACE output")

            self.bzeroflag = ace_section.bzeroflag 
            self.numtypes = ace_section.numtypes
            self.ranks = ace_section.ranks
            self.lmin = ace_section.lmin
            self.lmax = ace_section.lmax
            self.nmax = ace_section.nmax
            self.mumax = ace_section.mumax
            self.nmaxbase = ace_section.nmaxbase
            self.rcutfac = ace_section.rcutfac
            self.lmbda = ace_section.lmbda
            self.rcinner = ace_section.rcinner
            self.drcinner = ace_section.drcinner
            self.types = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))  # ACE uses 'types', PYACE uses 'elements'
            self.erefs = ace_section.erefs
            
            # Handle attributes that may not exist in PYACE
            self.bikflag = getattr(ace_section, 'bikflag', False)
            self.b_basis = getattr(ace_section, 'b_basis', 'pa_tabulated')
            self.wigner_flag = getattr(ace_section, 'wigner_flag', True)

            if self.bzeroflag:
                #assert len(self.types) ==  len(self.erefs), "must provide reference energy for each atom type"
                if len(self.types) ==  len(self.erefs):
                    reference_ens = [float(e0) for e0 in self.erefs]
                else:
                    reference_ens = [0.0] * len(self.types)
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


            ldict = {int(rank):int(lmax) for rank,lmax in zip(self.ranks,self.lmax)}
            L_R = 0
            M_R = 0
            rankstrlst = ['%s']*len(self.ranks)
            rankstr = ''.join(rankstrlst) % tuple(self.ranks)
            lstrlst = ['%s']*len(self.ranks)
            lstr = ''.join(lstrlst) % tuple(self.lmax)
            if not self.wigner_flag:
                try:
                    with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                        ccs = pickle.load(handle)
                except FileNotFoundError:
                    ccs = get_cg_coupling(ldict,L_R=L_R)
                    #print (ccs)
                    #store them for later so they don't need to be recalculated
                    store_generalized(ccs, coupling_type='cg',L_R=L_R)
            else:
                try:
                    with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                        ccs = pickle.load(handle)
                except FileNotFoundError:
                    ccs = get_wig_coupling(ldict,L_R)
                    #print (ccs)
                    #store them for later so they don't need to be recalculated
                    store_generalized(ccs, coupling_type='wig',L_R=L_R)


            apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, [int(k) for k in self.lmin], self.b_basis, **{'ccs':ccs[M_R]})
            if ace_section.bzeroflag:
                apot.set_betas(coeffs,has_zeros=False)
            else:
                apot.set_betas(coeffs,has_zeros=True)
            
            # Handle nus attribute - may not exist in PYACE
            if hasattr(ace_section, 'nus'):
                apot.set_funcs(nulst=ace_section.nus)
            
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

    # Support both ACE and PYACE sections
    if "ACE" in config.sections:
        desc_str = "ACE"
        ace_section = config.sections["ACE"]
    elif "PYACE" in config.sections:
        desc_str = "PYACE"
        ace_section = config.sections["PYACE"]
    else:
        raise RuntimeError("No ACE or PYACE section found for coefficient output")
    
    # Check if we have the required attributes for coefficient output
    if not hasattr(ace_section, 'blank2J') or not hasattr(ace_section, 'blist'):
        raise RuntimeError(f"Section {desc_str} missing required attributes 'blank2J' or 'blist' for coefficient output. PYACE coefficient output may not be supported.")
    
    coeffs = coeffs.reshape((ace_section.numtypes, -1))
    blank2Js = ace_section.blank2J.reshape((ace_section.numtypes, -1))
    if ace_section.bzeroflag:
        coeff_names = ace_section.blist
    else:
        coeff_names = [[0]]+ace_section.blist
    coeffs = np.multiply(coeffs, blank2Js)
    type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
    out = f"# FitSNAP generated on {datetime.now()} with Hash: {config.hash}\n\n"
    out += "{} {}\n".format(len(type_names), int(np.ceil(len(coeff_names)/ace_section.numtypes)))
    for elname, column in zip(type_names,
                        coeffs):
        out += "{}\n".format(elname)
        out += "\n".join(f" {bval:<30.18} #  B{bname} " for bval, bname in zip(column, coeff_names))
        out += "\n"
    out += "\n# End of potential"
    return out

def _to_potential_file(config):
    """
    Use config settings to write a LAMMPS potential .mod file.
    """
    
    # Support both ACE and PYACE sections
    if "ACE" in config.sections:
        ace_section = config.sections["ACE"]
    elif "PYACE" in config.sections:
        ace_section = config.sections["PYACE"]
    else:
        raise RuntimeError("No ACE or PYACE section found for potential file output")

    ps = config.sections["REFERENCE"].lmp_pairdecl[0]
    ace_filename = config.sections["OUTFILE"].potential_name.split("/")[-1]

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
        out += ps + " pace product\n"
        # add pair coeff commands from input, ignore if pair zero
        for pc in config.sections["REFERENCE"].lmp_pairdecl[1:]:
            out += f"{pc}\n" if "zero" not in pc else ""
        pc_ace = f"pair_coeff * * pace {ace_filename}.yace"
        type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
        for t in type_names:
            pc_ace += f" {t}"
        out += pc_ace
    else:
        out += "pair_style pace product\n"
        pc_ace = f"pair_coeff * * {ace_filename}.yace" 
        type_names = getattr(ace_section, 'types', getattr(ace_section, 'elements', ['H']))
        for t in type_names:
            pc_ace += f" {t}"
        out += pc_ace

    return out

