from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import DistributedList

import json, sys
import numpy as np
#from functools import reduce
from itertools import chain, groupby
from pprint import pprint

from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY

class LammpsReaxff(LammpsBase):

    # ----------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._lmp = None
        self.pt.check_lammps()

        self.potential = self.config.sections['REAXFF'].potential
        self.elements = self.config.sections['REAXFF'].elements
        self.masses = self.config.sections['REAXFF'].masses
        self.type_mapping = self.config.sections['REAXFF'].type_mapping

        self.charge_fix = self.config.sections['CALCULATOR'].charge_fix
        self._initialize_lammps()

    # ----------------------------------------------------------------

    def __del__(self):
        self._lmp = self.pt.close_lammps()
        del self

    # ----------------------------------------------------------------

    def process_configs_for_parameter_values(self, i, xi):

        try:
            self._lmp.set_reaxff_parameters(self.parameters, xi)
            self._lmp.command("run 0 post no")
            self._collect_lammps(i)
        except Exception as e:
            raise e



    # ----------------------------------------------------------------

    def allocate_per_config(self, data: list):

        np.set_printoptions(threshold=5, edgeitems=1)
        pprint(data, width=99, compact=True)
        len_data = len(data)
        len_all_data = self.pt.get_ncpn(len_data)
        popsize = self.config.sections['SOLVER'].popsize
        #self.pt.create_shared_array('weights', len_all_data, 1)
        #np.zeros((M, N, O))


        """
        if self.energy:
            #self.pt.create_shared_array('ground_index', len_all_data, 1, dtype='i')
            self.pt.create_shared_array('reference_energy', len_all_data, 1)
            self.pt.create_shared_array('predicted_energy', popsize, len_all_data)

        if self.energy:
            self.pt.create_shared_array('forces_reference', len_all_data, 1)
            self.pt.create_shared_array('forces_predicted', popsize, len_all_data)

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

    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------

    def set_data_index(self, data_index):

        self._data_index = data_index
        #pprint(self.pt.fitsnap_dict["Data"])
        self._data = self.pt.fitsnap_dict["Data"][data_index]
        self._lmp.command("delete_atoms group all")
        #print(f"self._data_index {self._data_index} self._data {self._data}")
        self._create_atoms_helper(type_mapping=self.type_mapping)


    # ----------------------------------------------------------------

    def _collect_lammps(self, i):

        #dist = self._lmp.numpy.extract_compute('dist',LMP_STYLE_LOCAL,LMP_TYPE_VECTOR)
        #q = self._lmp.numpy.extract_compute('charge',LMP_STYLE_ATOM,LMP_TYPE_VECTOR)

        if self.energy:
            pe = _extract_compute_np(self._lmp, 'thermo_pe',LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            self.pt.shared_arrays['predicted_energy'].array[i][self._data_index] = pe

        if self.force:
            f = self._lmp.extract_atom('f',LMP_STYLE_ATOM,LMP_TYPE_ARRAY)
            print(f"f {f}")
            self._data['predicted_forces'] = f

        if self.dipole:
            dipole = _extract_compute_np(self._lmp,'dipole',LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR)
            self._data['predicted_dipole'] = dipole
            #print(f"dipole {dipole}")

