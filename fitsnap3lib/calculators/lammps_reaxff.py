from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
import numpy as np

from pprint import pprint
import json

class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {'Charges': [0.0, 0.0, 0.0]}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

        self.force_field = self.config.sections['REAXFF'].force_field
        #print(self.force_field)

        self.parameters = eval(self.config.sections['REAXFF'].parameters)
        #pprint(self.parameters,width=150)


    # a array is for per-atom quantities in all configs (eg charge, ...)
    # b array is for per-config quantities like energy
    # c matrix is for per-atom 3-vectors like position and velocity.

    def get_width(self):
    #    a_width = self.config.sections["REAXFF"].ncoeff #+ 3
    #    return a_width
        return 0

    def _prepare_lammps(self):
        self._set_structure()

        # needs reworking when lammps will accept variable 2J
        #self._lmp.command(f"variable twojmax equal {max(self.config.sections['BISPECTRUM'].twojmax)}")

        #self._set_computes()
        #self._set_neighbor_list()
        self._lmp.command("dump 1 all custom 1 lammps.dump id x y z q")
        self._lmp.command("thermo 1")
        self._lmp.command("thermo_style custom step temp pe ke etotal press")
        self._lmp.command("pair_style reaxff NULL")
        self._lmp.command("pair_coeff * * reaxff-water/Water2017.ff H O X")
        self._lmp.command("fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff maxiter 400")


    def _set_box(self):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),(ay, by, cy),(az, bz, cz)) = self._data["Lattice"]
        self._lmp.command(f'region box block {-ax} {ax} {-by} {by} {-cz} {cz}')
        numtypes=self.config.sections['REAXFF'].numtypes
        self._lmp.command(f"create_box {numtypes} box")

    def _create_atoms(self):
        self._lmp.command("mass 1 1.0080")
        self._lmp.command("mass 2 15.9990")
        self._lmp.command("mass 3 1.0080")
        self._create_atoms_helper(type_mapping=self.config.sections["REAXFF"].type_mapping)

    def _set_computes(self):
        #self._lmp.command("compute reaxff_energy ")
        pass

    def _create_charge(self):
        pass

    def _collect_lammps(self):

        print("_collect_lammps(self)")

        #self._lmp.command("print $(c_thermo_pe)")
        self._lmp.command("print '$(c_thermo_pe) $(q[1]) $(q[2]) $(q[3])' append log.lammps")

        #self._lmp.command("info all out append lammps.txt")

        if self.config.sections["CALCULATOR"].energy:
            lmp_energy = _extract_compute_np(self._lmp, "thermo_pe", 0, 0)
            #print(lmp_energy)

