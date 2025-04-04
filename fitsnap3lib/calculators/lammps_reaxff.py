from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, sys
import numpy as np
#from functools import reduce
from itertools import chain, groupby
from pprint import pprint

from lammps import LAMMPS_DOUBLE, LAMMPS_DOUBLE_2D
from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

class LammpsReaxff(LammpsBase):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config)

        self.potential = self.config.sections["REAXFF"].potential
        self.elements = self.config.sections["REAXFF"].elements
        self.masses = self.config.sections["REAXFF"].masses
        self.type_mapping = self.config.sections["REAXFF"].type_mapping
        self.parameters = self.config.sections["REAXFF"].parameters
        self.charge_fix = self.config.sections["CALCULATOR"].charge_fix

        self.energy = self.config.sections["CALCULATOR"].energy
        self.force = self.config.sections["CALCULATOR"].force
        self.stress = self.config.sections["CALCULATOR"].stress
        self.charge = self.config.sections["CALCULATOR"].charge
        self.dipole = self.config.sections["CALCULATOR"].dipole

        self._lmp = None
        self.pt.check_lammps()

    # --------------------------------------------------------------------------------------------

    def __del__(self):

        self._lmp = self.pt.close_lammps()
        del self

    # --------------------------------------------------------------------------------------------

        #np.set_printoptions(threshold=5, edgeitems=1)
        #pprint(configs, width=99, compact=True)

    def allocate_per_config(self, configs: list):

        if self.pt.stubs == 0 and self.pt._rank == 0:
            ncpn = self.pt.get_ncpn(0)
            return

        self._configs = configs
        ncpn = self.pt.get_ncpn(len(configs))
        popsize = self.config.sections['SOLVER'].popsize
        if self.energy: self.sum_energy_residuals = np.zeros(popsize)
        if self.force: self.sum_forces_residuals = np.zeros(popsize)
        if self.charge: self.sum_charges_residuals = np.zeros(popsize)
        if self.dipole: self.sum_dipole_residuals = np.zeros(popsize)
        self.sum_residuals = np.zeros(popsize)

    # --------------------------------------------------------------------------------------------

    def process_configs_with_values(self, values):

        if self.energy: self.sum_energy_residuals[:] = 0.0
        if self.force: self.sum_forces_residuals[:] = 0.0
        if self.charge: self.sum_charges_residuals[:] = 0.0
        if self.dipole: self.sum_dipole_residuals[:] = 0.0
        self.sum_residuals[:] = 0.0

        for config_index, c in enumerate(self._configs):
            self._data = c

            for pop_index, v in enumerate(values):
                try:
                
                    if False:
                        logfile = f"{c['File']}".replace('/','').replace(' ','-')
                        with open(f"acks2/{logfile}.in","w") as f:
                            self._initialize_lammps(1,printfile=f)
                            self._lmp.command(f"variable config string {logfile}")
                            self._lmp.command("info variables")
                            self._prepare_lammps()
                            self._lmp.set_reaxff_parameters(self.parameters, v)
                            self._lmp.command("run 0 post no")
                            self._collect_lammps(config_index, pop_index)
                            self._lmp.command("unfix 1")
                            self._lmp.command("fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 1000")
                            self._lmp.command("run 0 post no")
                    else:
                        self._initialize_lammps(0, lammpsscreen=0)
                        self._prepare_lammps()
                        self._lmp.set_reaxff_parameters(self.parameters, v)
                        self._lmp.command("run 0 post no")
                        self._collect_lammps(config_index, pop_index)

                    self._lmp = self.pt.close_lammps()

                except Exception as e:
                    print(f"*** rank {self.pt._rank} exception {e}")
                    raise e

            #if self.force: self.sum_residuals += self.sum_forces_residuals

        #print(f"*** sum_energy_residuals {self.sum_energy_residuals}")
        #sum_forces_residuals {self.sum_forces_residuals} ")
        #if self.energy: self.sum_residuals += self._data["eweight"] * self.sum_energy_residuals
        #if self.force: self.sum_residuals += self._data["fweight"] * self.sum_forces_residuals
        #if self.charge: self.sum_residuals += self._data["cweight"] * self.sum_charges_residuals
        #if self.dipole: self.sum_residuals += self._data["dweight"] * self.sum_dipole_residuals

        #print(f"*** sum_residuals {self.sum_residuals} sum_energy_residuals {self.sum_energy_residuals}")
        if self.energy: self.sum_residuals += self.sum_energy_residuals
        if self.force: self.sum_residuals += self.sum_forces_residuals
        return self.sum_residuals

    # --------------------------------------------------------------------------------------------

    def _collect_lammps(self, config_index, pop_index):

        def pseudo_huber(x, delta=1.0):
            x = np.nan_to_num(x, nan=8e8)
            return delta**2 * (np.sqrt(1 + (x / delta)**2) - 1)

        def cauchy_loss(x, c=1.0):
            x = np.nan_to_num(x, nan=8e8)
            return c**2 * np.log1p((x / c)**2)

        def huber_loss(x, delta=1.0):
            x = np.nan_to_num(x, nan=8e8)
            abs_x = np.abs(x)
            return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))

        if self.energy:
            pe = self._lmp.get_thermo('pe')
            loss_energy = pseudo_huber(pe - self._data["Energy"], delta=1.0)
            energy_residual = self._data['eweight'] * loss_energy
            self.sum_energy_residuals[pop_index] += energy_residual

        if self.force:
            forces = self._lmp.numpy.extract_atom(
                name='f',
                dtype=LAMMPS_DOUBLE_2D,
                nelem=self._data["NumAtoms"],
                dim=3
            )
            loss_forces = pseudo_huber(forces - self._data["Forces"], delta=0.5)
            forces_residual = self._data['fweight'] * np.sum(loss_forces)
            self.sum_forces_residuals[pop_index] += forces_residual

        if self.charge:
            charges = self._lmp.numpy.extract_atom(
                name='q',
                dtype=LAMMPS_DOUBLE,
                nelem=self._data["NumAtoms"],
                dim=1
            )
            loss_charges = pseudo_huber(charges - self._data["Charges"], delta=0.05)
            charge_residual = self._data['cweight']*np.sum(loss_charges)
            self.sum_charges_residuals[pop_index] += charge_residual

        if self.dipole:
            dipole = self._lmp.numpy.extract_compute(
                'dipole',
                LMP_STYLE_GLOBAL,
                LMP_TYPE_VECTOR
            )
            loss_dipole = pseudo_huber(dipole - self._data["Dipole"], delta=0.1)
            dipole_residual = self._data['dweight']*np.sum(loss_dipole)
            self.sum_dipole_residuals[pop_index] += dipole_residual

        def signed_fmt(x, width=2, prec=0):
            if abs(x) < .01:
                return f"{0:>{width}}"
            elif x > 0:
                return f"+{x:>{width - 1}.{prec}f}"
            else:
                return f"{x:>{width}.{prec}f}"

        # print(f"*** rank {self.pt._rank} {self._data['File']:<12s} "
        #    f"({signed_fmt(np.sum(self._data['Charges']))}) pop_index {pop_index:<2} "
        #    f"| energy {energy_residual:12g} | force {forces_residual:12g} "
        #    f"| charge {charge_residual:12g} | dipole {dipole_residual:12g}")





    # --------------------------------------------------------------------------------------------

    def _prepare_lammps(self):

        self._lmp.command("clear")
        self._lmp.command("boundary f f f")
        reference = self.config.sections["REFERENCE"]
        if reference.units != "real" or reference.atom_style != "charge":
            raise NotImplementedError("FitSNAP-ReaxFF only supports 'units real' and 'atom_style charge'.")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")
        self._lmp.command(self._data["Region"])
        self._lmp.command(f"create_box {len(self.elements)} box")
        #self._lmp.command("delete_atoms group all")

        if True:
            for i in range(len(self.masses)): self._lmp.command(f"mass {i+1} {self.masses[i]}")
            types = [self.type_mapping[a_t] for a_t in self._data["AtomTypes"]]
            for t, p in zip(types, self._data["Positions"]):
                self._lmp.command(f"create_atoms {t} single {p[0]} {p[1]} {p[2]}")
        else:
            self._create_atoms_helper(type_mapping=self.type_mapping)
            self._lmp.commands_list([f"mass {i+1} {self.masses[i]}" for i in range(len(self.masses))])

        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command(f"pair_coeff * * {self.potential} {' '.join(self.elements)}")
        self._create_charge()
        sum_charges = round(np.sum(self._data["Charges"]))
        #self._lmp.command(self.charge_fix)
        self._lmp.command(self.charge_fix + f" target_charge {sum_charges}")
        if self.dipole: self._lmp.command("compute dipole all dipole")

    # --------------------------------------------------------------------------------------------



