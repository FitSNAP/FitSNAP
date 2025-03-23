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
        self._lmp = None
        self.pt.check_lammps()

        self.potential = self.config.sections['REAXFF'].potential
        self.elements = self.config.sections['REAXFF'].elements
        self.masses = self.config.sections['REAXFF'].masses
        self.type_mapping = self.config.sections['REAXFF'].type_mapping
        self.parameters = self.config.sections['REAXFF'].parameters
        self.charge_fix = self.config.sections['CALCULATOR'].charge_fix
        self._initialize_lammps()

    # --------------------------------------------------------------------------------------------

    def __del__(self):
        self._lmp = self.pt.close_lammps()
        del self

    # --------------------------------------------------------------------------------------------

    def process_configs_with_values(self, values):

        for i, c in enumerate(self._configs):
            self._data = c
            #print(f"*** rank {self.pt._rank} ok 1")
            self._lmp.command("delete_atoms group all")
            self._create_atoms_helper(type_mapping=self.type_mapping)

            for j, v in enumerate(values):
                try:
                    #print(f"*** rank {self.pt._rank} ok 4")
                    self._lmp.set_reaxff_parameters(self.parameters, v)
                    #print(f"*** rank {self.pt._rank} ok 5")
                    self._lmp.command("run 0 post no")
                    #print(f"*** rank {self.pt._rank} ok 6")
                    self._collect_lammps(i,j)
                    #print(f"*** rank {self.pt._rank} ok 7")
                except Exception as e:
                    raise e

    # --------------------------------------------------------------------------------------------

    def allocate_per_config(self, configs: list):

        len_configs = len(configs)
        self._configs = configs
        np.set_printoptions(threshold=5, edgeitems=1)
        #pprint(configs, width=99, compact=True)
        ncpn = self.pt.get_ncpn(len_configs)
        popsize = self.config.sections['SOLVER'].popsize
        #self.pt.create_shared_array('weights', ncpn, 1)
        #

        if self.energy:
            #self.pt.create_shared_array('energy', len_all_data, 1)
            #self.pt.create_shared_array('energy_predicted', popsize, len_all_data)
            self.eweight = configs[0].["eweight"]
            self.energy_predicted = np.zeros((popsize,len_configs))
            self.energy_reference = np.zeros((len_configs))
            for i, c in enumerate(configs):
                self.energy_reference[i] = c["Energy"]

        if self.force:
            self.fweight = configs[0].["fweight"]
            self.forces_predicted = np.zeros((popsize,len_configs,3))
            #self.pt.create_shared_array('forces_reference', len_all_data, 1)
            #self.pt.create_shared_array('forces_predicted', popsize, len_all_data)


    # --------------------------------------------------------------------------------------------

    def _initialize_lammps(self, printlammps=0):
        super()._initialize_lammps(printlammps=printlammps)
        self._lmp.command("clear")
        self._lmp.command("boundary p p p")

        reference = self.config.sections["REFERENCE"]
        if reference.units != "real" or reference.atom_style != "charge":
            raise NotImplementedError("FitSNAP-ReaxFF only supports 'units real' and 'atom_style charge'.")
        self._lmp.command("units real")
        self._lmp.command("atom_style charge")
        self._lmp.command("atom_modify map array sort 0 2.0")

        # FIXME
        #xlo, ylo, zlo = np.min(self._data["Positions"],axis=0)-10.0
        #xhi, yhi, zhi = np.max(self._data["Positions"],axis=0)+10.0
        #print(xlo, ylo, zlo, xhi, yhi, zhi)
        #self._lmp.command(f'region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}')
        self._lmp.command(f"region box block -15 15 -15 15 -15 15")

        self._lmp.command(f"create_box {len(self.elements)} box")
        self._lmp.commands_list([f"mass {i+1} {self.masses[i]}" for i in range(len(self.masses))])
        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command(f"pair_coeff * * {self.potential} {' '.join(self.elements)}")
        self._lmp.command(self.charge_fix)
        if self.dipole: self._lmp.command("compute dipole all dipole")

    # --------------------------------------------------------------------------------------------

    def _collect_lammps(self, config_index, pop_index):

        #dist = self._lmp.numpy.extract_compute('dist',LMP_STYLE_LOCAL,LMP_TYPE_VECTOR)
        #q = self._lmp.numpy.extract_compute('charge',LMP_STYLE_ATOM,LMP_TYPE_VECTOR)

        if self.energy:
            pe = _extract_compute_np(self._lmp, 'thermo_pe', LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
            print(f"*** rank {self.pt._rank} pe {pe}")
            self.energy_predicted[pop_index][config_index] = pe

        if self.force:
            f = self._lmp.extract_atom('f', LMP_STYLE_ATOM, LMP_TYPE_ARRAY)
            print(f"f {f}")
            self.forces_predicted[pop_index][config_index] = f

        if self.dipole:
            dipole = _extract_compute_np(self._lmp,'dipole',LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR)
            self._data['predicted_dipole'] = dipole
            #print(f"dipole {dipole}")

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
