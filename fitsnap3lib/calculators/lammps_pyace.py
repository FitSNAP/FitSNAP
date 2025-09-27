
import numpy as np
from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.calculators.lammps_pace import LammpsPace
import lammps

# Import pyace components directly to avoid circular imports
try:
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
    from pyace.asecalc import PyACECalculator
    from pyace import create_multispecies_basis_config
    from pyace.atomicenvironment import aseatoms_to_atomicenvironment
    from pyace.calculator import ACECalculator
    PYACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pyace: {e}")
    PYACE_AVAILABLE = False
    # Define dummy classes to prevent errors
    class PyACECalculator: pass
    class ACEBBasisSet: pass
    class ACECTildeBasisSet: pass
    class BBasisConfiguration: pass
    class ACECalculator: pass
    def create_multispecies_basis_config(*args, **kwargs): pass
    def aseatoms_to_atomicenvironment(*args, **kwargs): pass

# ------------------------------------------------------------------------------------------------

class LammpsPyace(LammpsPace):
    """
    Calculator using pyace basis in [PYACE] with LAMMPS compute pace
    """
    
    # --------------------------------------------------------------------------------------------

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, calculator_section="PYACE")
        
        self._data = {}
        self._i = 0
        self._row_index = 0
    
    # --------------------------------------------------------------------------------------------

    def get_width(self):
        """Get width of descriptor vector for PYACE calculator"""
        
        if not PYACE_AVAILABLE:
            raise RuntimeError("pyace not available")
        
        if self._bzeroflag:
            return self._ncoeff
        else:
            return self._ncoeff + self._numtypes
            
    # --------------------------------------------------------------------------------------------

    def _collect_lammps(self):
        num_atoms = self._data["NumAtoms"]
        n_coeff = self._ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids  = self._extract_atom_ids(num_atoms)
        lmp_pos  = self._extract_atom_positions(num_atoms)
        lmp_types  = self._extract_atom_types(num_atoms)
            
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms \nGroup and configuration: {} {}".format(self._data["Group"],self._data["File"])

        lmp_volume = self._lmp.get_thermo("vol")

        # Extract pace data, including reference potential data

        nrows_energy = 1
        bik_rows = 1
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_pace = nrows_energy + nrows_force + nrows_virial
        ncols_descriptors = self._ncoeff
        ncols_reference = 1
        ncols_pace = ncols_descriptors + ncols_reference
        index = self.shared_index
        dindex = self.distributed_index
        lmp_pace = _extract_compute_np(self._lmp, "pace", 0, 2, (nrows_pace, ncols_pace))

        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            self.pt.single_print('! WARNING! applying np.nan_to_num()')
            lmp_pace = np.nan_to_num(lmp_pace)
        if (np.isinf(lmp_pace)).any() or (np.isnan(lmp_pace)).any():
            raise ValueError('NaN in computed data of file {} in group {}'.format(self._data["File"], self._data["Group"]))

        irow = 0
        bik_rows = 1
        if self._bikflag:
            bik_rows = num_atoms
        icolref = ncols_descriptors
        if self.config.sections["CALCULATOR"].energy:
        
            b_sum_temp = lmp_pace[irow, :ncols_descriptors] / num_atoms

            if not self._bzeroflag:
                if self._bikflag:
                    raise NotImplementedError("Per atom energy is not implemented without bzeroflag")

                onehot_atoms = np.zeros(self._numtypes)
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[self._type_mapping[atom] - 1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=0)
        
            self.pt.shared_arrays['a'].array[index] = b_sum_temp
            ref_energy = lmp_pace[irow, icolref]
            self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if self.config.sections["CALCULATOR"].force:
            s = slice(index, index + num_atoms*ndim_force)
                        
            db_atom_temp = lmp_pace[irow:irow + nrows_force, :ncols_descriptors]
            db_atom_temp.shape = (num_atoms * ndim_force, self._ncoeff)

            if not self._bzeroflag:
                onehot_atoms = np.zeros((db_atom_temp.shape[0], self._numtypes))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=1)

            self.pt.shared_arrays['a'].array[s] = db_atom_temp
            ref_forces = lmp_pace[irow:irow + nrows_force, icolref]
            self.pt.shared_arrays['b'].array[s] = self._data["Forces"].ravel() - ref_forces
            self.pt.shared_arrays['w'].array[s] = self._data["fweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if self.config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_pace[irow:irow + nrows_virial, :ncols_descriptors] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, self._ncoeff * self._numtypes)
            if not self._bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], self._numtypes, self._ncoeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], self._numtypes, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], self._numtypes*(self._ncoeff+1))
            self.pt.shared_arrays['a'].array[index:index+ndim_virial] = vb_sum_temp
            ref_stress = lmp_pace[irow:irow + nrows_virial, icolref]
            self.pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            self.pt.shared_arrays['w'].array[index:index+ndim_virial] = self._data["vweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            index += ndim_virial
            dindex += ndim_virial

        length = dindex - self.distributed_index

        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
        self.shared_index = index
        self.distributed_index = dindex
    
    # --------------------------------------------------------------------------------------------


