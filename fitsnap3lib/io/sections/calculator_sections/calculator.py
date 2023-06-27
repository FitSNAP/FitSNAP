from fitsnap3lib.io.sections.sections import Section
#from fitsnap3lib.parallel_tools import ParallelTools


#pt = ParallelTools()


class Calculator(Section):

    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        #self.pt = ParallelTools()
        self.allowedkeys = ['calculator', 'energy', 'per_atom_energy', 'force', 'stress', \
                            'nonlinear', 'per_atom_scalar']
        self._check_section()

        self.calculator = self.get_value("CALCULATOR", "calculator", "LAMMPSSNAP")
        self.energy = self.get_value("CALCULATOR", "energy", "True", "bool")
        self.per_atom_energy = self.get_value("CALCULATOR", "per_atom_energy", "False", "bool")
        self.per_atom_scalar = self.get_value("CALCULATOR", "per_atom_scalar", "False", "bool")

        self.dee = self.check_path(self.get_value("CALCULATOR", "dee", "detailed_energy_errors.dat"))
        self.pt.add_2_fitsnap("energy", self.energy)
        self.pt.add_2_fitsnap("per_atom_energy", self.per_atom_energy)
        self.pt.add_2_fitsnap("per_atom_scalar", self.per_atom_scalar)
        self.force = self.get_value("CALCULATOR", "force", "True", "bool")
        self.dfe = self.check_path(self.get_value("CALCULATOR", "dfe", "detailed_force_errors.dat"))
        self.pt.add_2_fitsnap("force", self.force)
        self.stress = self.get_value("CALCULATOR", "stress", "True", "bool")
        self.dse = self.check_path(self.get_value("CALCULATOR", "dee", "detailed_stress_errors.dat"))
        self.pt.add_2_fitsnap("stress", self.stress)
        self.nonlinear = self.get_value("CALCULATOR", "nonlinear", "False", "bool")

        self.pt.add_2_fitsnap("nonlinear", self.nonlinear)


        if (self.nonlinear):
            self.linear = False
        else:
            self.linear = True

        if (self.per_atom_scalar and (self.force or self.energy)):
            raise Exception("Cannot fit to arbitrary per-atom scalars and forces/energies.")
        if (self.per_atom_scalar and self.linear):
            raise Exception("Fitting to arbitrary per-atom scalars only implemented for nonlinear solvers.")
            
        self.delete()
