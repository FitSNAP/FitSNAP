
import numpy as np
from fitsnap3lib.calculators.lammps_base import LammpsBase
import lammps

# Avoid circular import by importing pyace components when needed
# Will import PyACECalculator, ACEBBasisSet, ACECTildeBasisSet in setup_pyace()


class LammpsPyACE(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)





class LammpsPyace(LammpsBase):
    """
    Calculator using pyace for ACE descriptor calculations
    instead of LAMMPS compute pace
    """
    
    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        
        self._data = {}
        self._i = 0
        self._row_index = 0
        
        # Initialize pyace specific parameters
        self.ace_basis = None
        self.pyace_calc = None
        self.setup_pyace()
        
        
    def get_width(self):
    
        # FIXME
    
        if (self.config.sections["CALCULATOR"].nonlinear):
            a_width = self.config.sections["ACE"].ncoeff
        else:
            num_types = self.config.sections["ACE"].numtypes
            a_width = self.config.sections["ACE"].ncoeff  * num_types
            if not self.config.sections["ACE"].bzeroflag:
                a_width += num_types
        return a_width

    
    def setup_pyace(self):
        """Initialize pyace calculator with basis functions"""
        
        # Get ACE configuration from FitSNAP config
        ace_config = self.config.sections["ACE"]
        
        # Load basis set - can be .yaml or .yace format
        basis_file = ace_config.get("ccs_file", None)
        
        if basis_file:
            if basis_file.endswith('.yaml'):
                # B-basis format
                self.ace_basis = ACEBBasisSet(basis_file)
            elif basis_file.endswith('.yace'):
                # Ctilde-basis format
                self.ace_basis = ACECTildeBasisSet(basis_file)
            
            # Create pyace calculator
            self.pyace_calc = PyACECalculator(self.ace_basis)
    
    def calculate_descriptors(self, data):
        """
        Calculate ACE descriptors using pyace instead of LAMMPS compute
        """
        
        # Convert LAMMPS atoms to ASE atoms for pyace
        ase_atoms = self._lammps_to_ase(data)
        
        # Set calculator
        ase_atoms.set_calculator(self.pyace_calc)
        
        # Get descriptors directly from pyace
        # This replaces the LAMMPS compute pace call
        descriptors = self._get_ace_descriptors(ase_atoms)
        
        # Store in FitSNAP format
        self._store_descriptors(descriptors, data)
        
        return descriptors
    
    def _get_ace_descriptors(self, atoms):
        """
        Extract ACE descriptors from pyace calculator
        """
        
        # Calculate properties
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Get per-atom descriptors if available
        if hasattr(self.pyace_calc, 'get_descriptors'):
            descriptors = self.pyace_calc.get_descriptors(atoms)
        else:
            # Fallback to computing from basis functions
            descriptors = self._compute_descriptors_from_basis(atoms)
        
        return {
            'energy': energy,
            'forces': forces,
            'descriptors': descriptors
        }
    
    def _compute_descriptors_from_basis(self, atoms):
        """
        Compute ACE descriptors from basis functions
        """
        
        # This would use the ACE basis functions directly
        # Implementation depends on pyace version and features
        
        n_atoms = len(atoms)
        n_descriptors = self.ace_basis.get_number_of_functions()
        
        descriptors = np.zeros((n_atoms, n_descriptors))
        
        # Calculate ACE descriptors for each atom
        for i in range(n_atoms):
            # Get local environment
            neighbors = self._get_neighbors(atoms, i)
            
            # Evaluate basis functions
            descriptors[i] = self.ace_basis.evaluate(neighbors)
        
        return descriptors
    
    def _lammps_to_ase(self, data):
        """Convert LAMMPS data to ASE atoms object"""
        
        from ase import Atoms
        
        # Extract positions, types, cell from LAMMPS data
        positions = data['Positions']
        types = data['AtomTypes']
        cell = data['Lattice']
        
        # Map LAMMPS types to chemical symbols
        symbols = [self.config.sections["ACE"]["type_mapping"][t] for t in types]
        
        # Create ASE atoms
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True
        )
        
        return atoms
        

