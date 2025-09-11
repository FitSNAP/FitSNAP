
import numpy as np
from fitsnap3lib.calculators.lammps_base import LammpsBase
import lammps

# Import pyace components directly to avoid circular imports
try:
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet
    from pyace.calculators import PyACECalculator
    PYACE_AVAILABLE = True
except ImportError:
    try:
        # Alternative import paths
        import pyace.basis as pyace_basis
        import pyace.calculators as pyace_calc
        ACEBBasisSet = pyace_basis.ACEBBasisSet
        ACECTildeBasisSet = pyace_basis.ACECTildeBasisSet
        PyACECalculator = pyace_calc.PyACECalculator
        PYACE_AVAILABLE = True
    except ImportError:
        try:
            # Try importing individual components
            from pyace import PyACECalculator
            from pyace import ACEBBasisSet, ACECTildeBasisSet
            PYACE_AVAILABLE = True
        except ImportError as e:
            print(f"Warning: Could not import pyace: {e}")
            PYACE_AVAILABLE = False
            # Define dummy classes to prevent errors
            class PyACECalculator: pass
            class ACEBBasisSet: pass
            class ACECTildeBasisSet: pass


class LammpsPyACE(LammpsBase):
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
        """Get width of descriptor vector for PYACE calculator"""
        
        if not PYACE_AVAILABLE:
            self.pt.single_print("Warning: pyace not available, using fallback width calculation")
            return self._fallback_width_calculation()
        
        # If pyace basis is loaded, get actual width from pyace
        if self.ace_basis is not None:
            try:
                # Get exact number of basis functions from pyace
                if hasattr(self.ace_basis, 'get_number_of_functions'):
                    ncoeff = self.ace_basis.get_number_of_functions()
                elif hasattr(self.ace_basis, 'get_basis_size'):
                    ncoeff = self.ace_basis.get_basis_size()
                elif hasattr(self.ace_basis, 'total_number_of_functions'):
                    ncoeff = self.ace_basis.total_number_of_functions
                elif hasattr(self.ace_basis, 'basis_size'):
                    ncoeff = self.ace_basis.basis_size
                else:
                    # Try to get from pyace calculator if basis doesn't have the method
                    if self.pyace_calc is not None and hasattr(self.pyace_calc, 'get_number_of_basis_functions'):
                        ncoeff = self.pyace_calc.get_number_of_basis_functions()
                    else:
                        raise AttributeError("Cannot find method to get basis function count")
                        
            except Exception as e:
                self.pt.single_print(f"Warning: Could not get basis size from pyace: {e}")
                return self._fallback_width_calculation()
        else:
            # No basis loaded yet, use fallback
            return self._fallback_width_calculation()
        
        # Apply FitSNAP width calculation logic
        if (self.config.sections["CALCULATOR"].nonlinear):
            a_width = ncoeff
        else:
            num_types = self.config.sections["PYACE"].numtypes
            a_width = ncoeff * num_types
            if not self.config.sections["PYACE"].bzeroflag:
                a_width += num_types
        return a_width
    
    def _fallback_width_calculation(self):
        """Fallback width calculation when pyace is not available or basis not loaded"""
        # Use basic ACE calculation for backwards compatibility
        pyace_config = self.config.sections["PYACE"]
        
        if hasattr(pyace_config, 'ranks') and hasattr(pyace_config, 'nmax'):
            # Simple count based on ranks and nmax
            ranks = pyace_config.ranks if hasattr(pyace_config.ranks, '__iter__') else [pyace_config.ranks]
            nmax_vals = pyace_config.nmax if hasattr(pyace_config.nmax, '__iter__') else [pyace_config.nmax]
            
            num_types = len(pyace_config.elements)
            total_funcs = sum(int(nmax) * num_types for nmax in nmax_vals)
            
            return total_funcs
        else:
            # Minimal fallback
            return len(pyace_config.elements) * 10  # Conservative estimate
    

    def setup_pyace(self):
        """Initialize pyace calculator with basis functions"""
        
        if not PYACE_AVAILABLE:
            self.pt.single_print("Warning: pyace not available, cannot setup pyace calculator")
            self.ace_basis = None
            self.pyace_calc = None
            return
        
        # Get PYACE configuration
        pyace_config = self.config.sections["PYACE"]
        
        # Get the ACE configuration dict from the section
        if hasattr(pyace_config, 'get_ace_config'):
            ace_config_dict = pyace_config.get_ace_config()
        else:
            # Build config dict from section attributes
            ace_config_dict = {
                'cutoff': getattr(pyace_config, 'cutoff', 10.0),
                'deltaSplineBins': getattr(pyace_config, 'delta_spline_bins', 0.001),
                'elements': getattr(pyace_config, 'elements', ['H']),
                'embeddings': getattr(pyace_config, 'embeddings', {}),
                'bonds': getattr(pyace_config, 'bonds', {}),
                'functions': getattr(pyace_config, 'functions', {})
            }
        
        try:
            # Create pyace basis from configuration using the imported pyace classes
            self.ace_basis = self._create_basis_from_config(ace_config_dict)
            
            # Create pyace calculator
            if self.ace_basis is not None:
                self.pyace_calc = PyACECalculator(self.ace_basis)
                self.pt.single_print(f"Successfully created pyace calculator with {self.get_width()} basis functions")
            else:
                self.pt.single_print("Warning: Could not create pyace basis, calculator not initialized")
                
        except Exception as e:
            self.pt.single_print(f"Warning: Error setting up pyace calculator: {e}")
            self.ace_basis = None
            self.pyace_calc = None
    
    def _create_basis_from_config(self, config_dict):
        """Create pyace basis from configuration dictionary"""
        try:
            # Use pyace to create basis from configuration
            # This depends on the specific pyace API for your version
            
            # Try different pyace basis creation methods
            if hasattr(ACEBBasisSet, 'from_config'):
                # Method 1: Direct config creation
                return ACEBBasisSet.from_config(config_dict)
            elif hasattr(ACEBBasisSet, 'from_dict'):
                # Method 2: From dictionary
                return ACEBBasisSet.from_dict(config_dict)
            else:
                # Method 3: Try creating with constructor parameters
                # Extract key parameters
                elements = config_dict.get('elements', ['H'])
                cutoff = config_dict.get('cutoff', 10.0)
                embeddings = config_dict.get('embeddings', {})
                bonds = config_dict.get('bonds', {})
                functions = config_dict.get('functions', {})
                
                # Try to create basis with available constructor
                try:
                    return ACEBBasisSet(
                        elements=elements,
                        cutoff=cutoff,
                        embeddings=embeddings,
                        bonds=bonds,
                        functions=functions
                    )
                except TypeError:
                    # Constructor doesn't accept these parameters
                    # Try with minimal parameters
                    return ACEBBasisSet(elements=elements, cutoff=cutoff)
                    
        except Exception as e:
            self.pt.single_print(f"Error creating pyace basis from config: {e}")
            # Try alternative approaches
            try:
                # Fallback: try creating with just elements
                elements = config_dict.get('elements', ['H'])
                return ACEBBasisSet(elements)
            except Exception as e2:
                self.pt.single_print(f"Fallback basis creation also failed: {e2}")
                return None
    
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
        symbols = [self.config.sections["PYACE"]["type_mapping"][t] for t in types]
        
        # Create ASE atoms
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True
        )
        
        return atoms
        

