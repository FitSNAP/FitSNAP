from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, sys
import numpy as np
#from functools import reduce
from itertools import chain, groupby
from pprint import pprint

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
        self._initialize_lammps(0)

    # --------------------------------------------------------------------------------------------

    def __del__(self):

        self._lmp = self.pt.close_lammps()
        del self

    # --------------------------------------------------------------------------------------------

    def allocate_per_config(self, configs: list):

        if self.pt._rank == 0:
            ncpn = self.pt.get_ncpn(0)
            return

        len_configs = len(configs)
        self._configs = configs
        #np.set_printoptions(threshold=5, edgeitems=1)
        #pprint(configs, width=99, compact=True)
        ncpn = self.pt.get_ncpn(len_configs)
        popsize = self.config.sections['SOLVER'].popsize
        #self.pt.create_shared_array('weights', ncpn, 1)

        self.sum_energy_residuals = np.zeros(popsize)

        if self.energy:
            #self.pt.create_shared_array('energy', len_all_data, 1)
            self.eweight = configs[0]["eweight"]
            self.energy_predicted = np.zeros((popsize,len_configs))
            self.energy_reference = np.zeros((len_configs))
            for i, c in enumerate(configs):
                self.energy_reference[i] = c["Energy"]

        if self.force:
            self.fweight = configs[0]["fweight"]
            self.forces_predicted = np.zeros((popsize,len_configs,3))

    # --------------------------------------------------------------------------------------------

    def process_configs_with_values(self, values):

        self.sum_energy_residuals[:] = 0

        for config_index, c in enumerate(self._configs):
            self._data = c
            #print(f"*** rank {self.pt._rank} ok 1b")
            self._prepare_lammps()

            for pop_index, v in enumerate(values):
                try:
                    #print(f"*** rank {self.pt._rank} ok 1c")
                    self._lmp.set_reaxff_parameters(self.parameters, v)
                    self._lmp.command("run 0 post no")
                    self._collect_lammps(config_index,pop_index)

                except Exception as e:
                    print(f"*** rank {self.pt._rank} exception {e}")
                    raise e

        return self.sum_energy_residuals

    # --------------------------------------------------------------------------------------------

    def _prepare_lammps(self):
        self._lmp.command("clear")
        self._lmp.command("boundary p p p")

        reference = self.config.sections["REFERENCE"]
        if reference.units != "real" or reference.atom_style != "charge":
            raise NotImplementedError("FitSNAP-ReaxFF only supports 'units real' and 'atom_style charge'.")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")

        # FIXME
        xlo, ylo, zlo = np.min(self._data["Positions"],axis=0)-10.0
        xhi, yhi, zhi = np.max(self._data["Positions"],axis=0)+10.0
        #print(xlo, ylo, zlo, xhi, yhi, zhi)
        self._lmp.command(f'region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}')
        #self._lmp.command(f"region box block -15 15 -15 15 -15 15")

        self._lmp.command(f"create_box {len(self.elements)} box")
        #self._lmp.command("delete_atoms group all")
        self._create_atoms_helper(type_mapping=self.type_mapping)
        self._lmp.commands_list([f"mass {i+1} {self.masses[i]}" for i in range(len(self.masses))])
        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command(f"pair_coeff * * {self.potential} {' '.join(self.elements)}")
        self._create_charge()
        self._lmp.command(self.charge_fix)
        if self.dipole: self._lmp.command("compute dipole all dipole")

    # --------------------------------------------------------------------------------------------

    def _collect_lammps(self, config_index, pop_index):

        if self.energy:
            pe = _extract_compute_np(self._lmp, 'thermo_pe', LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
            #print(f"*** rank {self.pt._rank} config_index {config_index} e_ref {self.energy_reference[config_index]} pe {pe}")
            self.energy_predicted[pop_index][config_index] = pe

            energy_residual = pe - self.energy_reference[config_index]
            self.sum_energy_residuals[pop_index] += np.square(energy_residual)

        #    return np.sum(weights * np.nan_to_num(



            #if np.isnan(pe):
            #    print(f"*** rank {self.pt._rank} config {self._data}")


        if self.force:
            forces = self._lmp.extract_atom('f', LMP_STYLE_ATOM, LMP_TYPE_ARRAY)
            print(f"*** rank {self.pt._rank} forces {forces}")
            self.forces_predicted[pop_index][config_index] = forces

        if self.charge:
            charges = self._lmp.extract_atom('q', LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            print(f"*** rank {self.pt._rank} charges {charges}")
            self.charges_predicted[pop_index][config_index] = charges

        if self.dipole:
            dipole = _extract_compute_np(self._lmp, 'dipole', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
            print(f"*** rank {self.pt._rank} charges {charges}")
            self.dipole_predicted[pop_index][config_index] = dipole

    # --------------------------------------------------------------------------------------------






















################################ SCRATCH ################################

"""

        #data = sorted(data, key=keyfunc)

        i=0

        for k, g in groupby(all_data, lambda d: d["Group"]):
            group=list(g)
            ground_index = 0
            ground_energy = 999999.99

            for j, d in enumerate(group):

                # FIXME: let users choose manual weights
                #if "Weight" not in d: d["Weight"] = 1.0

                if d["Energy"] < ground_energy:
                    ground_index, ground_energy = j, d["Energy"]

            for j, d in enumerate(group):
                self.pt.shared_arrays['ground_index'].array[i+j] = i + ground_index
                self.pt.shared_arrays['reference_energy'].array[i+j] = d["Energy"] - ground_energy

            qm_y = self.pt.shared_arrays['reference_energy'].array[i:i+len(group)]
            auto_weights = np.square(np.max(qm_y)*1.1-np.array(qm_y))
            self.pt.shared_arrays['weights'].array[i:i+len(group)] = auto_weights/np.sum(auto_weights)
            i+=len(group)

        #if self.force: self.pt.create_shared_array('predicted_dipole', len_all_data, 1)
        if self.stress: raise NotImplementedError("FitSNAP-ReaxFF does not support stress fitting.")
        #if self.dipole: self.pt.create_shared_array('predicted_dipole', len_all_data, 1)

        if(False and self.pt._rank==0):

            print(f"*** [rank {self.pt._rank}] ground_index {self.pt.shared_arrays['ground_index'].array}")

            print(f"*** [rank {self.pt._rank}] qm_y {qm_y}")

            print(f"*** [rank {self.pt._rank}] reference_energy {self.pt.shared_arrays['reference_energy'].array}")

            print(f"*** [rank {self.pt._rank}] reference_energy {self.pt.shared_arrays['reference_energy'].array}")
"""
