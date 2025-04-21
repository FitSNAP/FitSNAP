from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList
import json, sys, gc
import numpy as np
from ctypes import c_int, c_double
from lammps import LAMMPS_DOUBLE, LAMMPS_DOUBLE_2D
from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

class LammpsReaxff(LammpsBase):

    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):

        super().__init__(name, pt, config)

        self.potential = config.sections["REAXFF"].potential
        self.elements = config.sections["REAXFF"].elements
        self.masses = config.sections["REAXFF"].masses
        self.type_mapping = config.sections["REAXFF"].type_mapping
        self.parameters = config.sections["REAXFF"].parameters
        self.charge_fix = config.sections["CALCULATOR"].charge_fix

        self.energy = config.sections["CALCULATOR"].energy
        self.force = config.sections["CALCULATOR"].force
        self.stress = config.sections["CALCULATOR"].stress
        self.charge = config.sections["CALCULATOR"].charge
        self.dipole = config.sections["CALCULATOR"].dipole
        self.quadrupole = config.sections["CALCULATOR"].quadrupole
        self.bond_order = config.sections["CALCULATOR"].bond_order

        self._lmp = None
        self.pt.check_lammps()
        #f = open(f"water.in","w")
        self._initialize_lammps(printlammps=0, lammpsscreen=0) # printfile=f

    # --------------------------------------------------------------------------------------------

    def __del__(self):

        if hasattr(self, '_lmp') and self._lmp is not None:
            self.pt.close_lammps()
            del self._lmp
            gc.collect()

    # --------------------------------------------------------------------------------------------

    def process_configs_with_values(self, values):

        if self.energy: self.sum_energy_residuals = 0.0
        if self.force: self.sum_force_residuals = 0.0
        if self.charge: self.sum_charge_residuals = 0.0
        if self.dipole: self.sum_dipole_residuals = 0.0
        if self.quadrupole: self.sum_quadrupole_residuals = 0.0
        if self.bond_order: self.bond_order_residuals = 0.0
        self._lmp.set_reaxff_parameters(self.parameters, values)
        self._charge_fix()

        for config_index, c in enumerate(self._configs):
            self._data = c
            self._prepare_lammps()
            try:
                self._run_lammps()
                self._collect_lammps(config_index)
            except Exception as e:
                print(f"*** rank {self.pt._rank} exception {e}")
                raise e

        self._lmp.command(f"unfix {self.charge_fix.split()[1]}")
        answer = []
        if self.energy: answer.append(self.sum_energy_residuals)
        if self.force: answer.append(self.sum_force_residuals)
        if self.charge: answer.append(self.sum_charge_residuals)
        if self.dipole: answer.append(self.sum_dipole_residuals)
        if self.quadrupole: answer.append(self.sum_quadrupole_residuals)
        if self.bond_order: answer.append(self.sum_bond_order_residuals)
        answer.append(sum(answer))
        return np.array(answer)

    # --------------------------------------------------------------------------------------------

    def _charge_fix(self):
        #sum_charges = round(np.sum(self._data["Charges"]))
        sum_charges = 0.0
        #self._lmp.command(self.charge_fix)
        self._lmp.command(self.charge_fix + f" target_charge {sum_charges}")

    # --------------------------------------------------------------------------------------------

    def _initialize_lammps(self, **kwargs):

        super()._initialize_lammps(**kwargs)
        self._lmp.command("boundary f f f")
        reference = self.config.sections["REFERENCE"]
        if reference.units != "real" or reference.atom_style != "charge":
            raise NotImplementedError("FitSNAP-ReaxFF only supports 'units real' and 'atom_style charge'.")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")
        self._lmp.command("region box block -99 99 -99 99 -99 99")
        self._lmp.command(f"create_box {len(self.elements)} box")
        for i in range(len(self.masses)): self._lmp.command(f"mass {i+1} {self.masses[i]}")
        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command(f"pair_coeff * * {self.potential} {' '.join(self.elements)}")

    # --------------------------------------------------------------------------------------------

        #np.set_printoptions(threshold=5, edgeitems=1)
        #pprint(configs, width=99, compact=True)

    def allocate_per_config(self, configs: list):

        self._configs = configs
        ncpn = self.pt.get_ncpn(len(configs))
        if self.dipole: self._lmp.command("compute dipole all dipole fixedorigin")
        if self.quadrupole: self._lmp.command("compute quadrupole all quadrupole")

        max_atoms = max(c["NumAtoms"] for c in configs)
        self._lmp.create_atoms( n=max_atoms, id=None,
            type=(max_atoms * c_int)(*([1] * max_atoms)),
            x=(max_atoms * 3 * c_double)(*([0.0] * max_atoms * 3))
        )

        self._nlocal = self._lmp.extract_global("nlocal")
        self._boxlo = self._lmp.extract_global("boxlo")
        self._boxhi = self._lmp.extract_global("boxhi")
        self._sublo = self._lmp.extract_global("sublo")
        self._subhi = self._lmp.extract_global("subhi")
        self._type = self._lmp.extract_atom("type")
        self._x = self._lmp.extract_atom("x")
        self._q = self._lmp.extract_atom("q")

    # --------------------------------------------------------------------------------------------

    def _prepare_lammps(self):

      self._nlocal = self._data["NumAtoms"]
      self._boxlo[:] = self._data["Bounds"][0]
      self._boxhi[:] = self._data["Bounds"][1]
      self._sublo[:] = self._data["Bounds"][0]
      self._subhi[:] = self._data["Bounds"][1]

      for i, (x, y, z) in enumerate(self._data["Positions"]):
          self._type[i] = self.type_mapping[self._data["AtomTypes"][i]]
          self._x[i][0], self._x[i][1], self._x[i][2] = x, y, z
          self._q[i] = self._data["Charges"][i][0]

    # --------------------------------------------------------------------------------------------

    def _collect_lammps(self, config_index):

        def pseudo_huber(x, delta=1.0):
            x = np.nan_to_num(x, nan=8e8)
            return delta**2 * (np.sqrt(1 + (x / delta)**2) - 1)
            #return x

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
            self.sum_energy_residuals += energy_residual

        if self.force:
            forces = self._lmp.numpy.extract_atom(
                name='f',
                dtype=LAMMPS_DOUBLE_2D,
                nelem=self._data["NumAtoms"],
                dim=3
            )
            loss_force = pseudo_huber(forces - self._data["Forces"], delta=0.5)
            force_residual = self._data['fweight'] * np.sum(loss_force)
            self.sum_force_residuals += force_residual

        if self.charge:
            charges = self._lmp.numpy.extract_atom(
                name='q',
                dtype=LAMMPS_DOUBLE,
                nelem=self._data["NumAtoms"],
                dim=1
            )
            loss_charge = pseudo_huber(charges - self._data["Charges"], delta=0.05)
            charge_residual = self._data['cweight']*np.sum(loss_charge)
            self.sum_charge_residuals += charge_residual

        if self.dipole:
            dipole = self._lmp.numpy.extract_compute('dipole', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            loss_dipole = pseudo_huber(dipole - self._data["Dipole"], delta=0.1)
            dipole_residual = self._data['dweight']*np.sum(loss_dipole)
            #print(f"*** dipole {dipole} loss_dipole {loss_dipole} dipole_residual {dipole_residual}")
            self.sum_dipole_residuals += dipole_residual

        if self.quadrupole:
            quadrupole = self._lmp.numpy.extract_compute('quadrupole', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            Q = self._data["Quadrupole"] # Q_xx Q_yy Q_zz Q_xy Q_xz Q_yz
            Q_ref = np.array([Q[0, 0], Q[1, 1], Q[2, 2], Q[0, 1], Q[0, 2], Q[1, 2]], dtype=np.float64)
            loss_quadrupole = pseudo_huber(quadrupole - Q_ref, delta=0.1)
            quadrupole_residual = self._data['qweight'] * np.sum(loss_quadrupole)
            #print(f"*** quadrupole {quadrupole} loss_quadrupole {loss_quadrupole} quadrupole_residual {quadrupole_residual}")
            self.sum_quadrupole_residuals += quadrupole_residual

        def signed_fmt(x, width=2, prec=0):
            if abs(x) < .01:
                return f"{0:>{width}}"
            elif x > 0:
                return f"+{x:>{width - 1}.{prec}f}"
            else:
                return f"{x:>{width}.{prec}f}"

        # pop_index {pop_index:<3}

        #print(f"*** rank {self.pt._rank} {self._data['File']:<12s} "
        #    f"({signed_fmt(np.sum(self._data['Charges']))})  "
        #    f"| energy {energy_residual:12g} | force {force_residual:12g} "
        #    f"| charge {charge_residual:12g} "
        #    f"| dipole {dipole_residual:12g} | quadrupole {quadrupole_residual:12g}")


    # --------------------------------------------------------------------------------------------

