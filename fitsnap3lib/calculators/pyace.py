
import numpy as np
from fitsnap3lib.calculators.calculator import Calculator
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


class PyACE(Calculator):
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
            raise RuntimeError("pyace not available")
        
        # If pyace basis is loaded, get actual width from pyace by computing a test case
        if self.ace_basis is not None:
            try:
                # Get actual width by creating a simple test case and computing descriptors
                # This ensures we get the exact width that PyACE will actually compute
                from ase import Atoms
                import numpy as np
                
                # Create a simple single-atom test case
                pyace_config = self.config.sections["PYACE"]
                test_atoms = Atoms(
                    symbols=[pyace_config.elements[0]], 
                    positions=[[0, 0, 0]], 
                    cell=[10, 10, 10], 
                    pbc=True
                )
                
                # Convert to atomic environment
                cutoff = self.ace_basis.cutoffmax
                elements_mapper_dict = {el: i for i, el in enumerate(self.ace_basis.elements_name)}
                atomic_env = aseatoms_to_atomicenvironment(
                    test_atoms, 
                    cutoff=cutoff,
                    elements_mapper_dict=elements_mapper_dict
                )
                
                # Create ACECalculator and compute descriptors to get actual width
                ace_calc = ACECalculator()
                if hasattr(self.ace_basis, 'evaluator'):
                    ace_calc.set_evaluator(self.ace_basis.evaluator)
                else:
                    from pyace.evaluator import ACEBEvaluator
                    evaluator = ACEBEvaluator(self.ace_basis)
                    ace_calc.set_evaluator(evaluator)
                
                # Compute projections to get actual width
                ace_calc.compute(atomic_env, compute_projections=True)
                projections = np.array(ace_calc.projections)
                
                # Get width from projections shape
                if len(projections.shape) == 1:
                    actual_width = len(projections) // atomic_env.n_atoms_real
                else:
                    actual_width = projections.shape[1]
                
                return actual_width
                        
            except Exception as e:
                raise e
        else:
            # No basis loaded yet, use fallback
            raise RuntimeError("self.ace_basis is None.")

    def setup_pyace(self):
        """Initialize pyace calculator with basis functions"""
        
        if not PYACE_AVAILABLE:
            self.pt.single_print("Warning: pyace not available, cannot setup pyace calculator")
            self.ace_basis = None
            self.pyace_calc = None
    
    def process_configs(self, data, i):
        """
        Calculate ACE descriptors for a given configuration using PyACE.
        This replaces LAMMPS-based computation with direct PyACE calls.
        
        Args:
            data: dictionary containing structural and fitting info for a configuration
            i: integer index for the configuration
        """
        
        if not PYACE_AVAILABLE:
            raise RuntimeError("PyACE not available - cannot process configs")
            
        if self.ace_basis is None or self.pyace_calc is None:
            raise RuntimeError("PyACE calculator not properly initialized")
        
        try:
            # Store data and index
            self._data = data
            self._i = i
            
            #self.pt.single_print(f"DEBUG: Processing config {i} with {len(data['Positions'])} atoms")
            
            # Convert FitSNAP data to ASE atoms (or use existing ase_atoms if available)
            # This avoids recreating ASE atoms when they're already stored (e.g., from pacemaker scraper)
            ase_atoms = self._fitsnap_data_to_ase(data)
            
            # Get atomic environment for PyACE
            atomic_env = self._get_atomic_environment(ase_atoms)
            
            # Compute ACE descriptors and derivatives using PyACE
            descriptors_data = self._compute_ace_descriptors(atomic_env)
            
            # Store results in FitSNAP shared arrays format
            self._store_fitsnap_results(descriptors_data, data)
            
        except Exception as e:
            self.pt.single_print(f"ERROR in process_configs for config {i}: {e}")
            import traceback
            self.pt.single_print(f"ERROR traceback: {traceback.format_exc()}")
            raise e
    
    def _fitsnap_data_to_ase(self, data):
        """
        Convert FitSNAP data format to ASE Atoms object
        """
        from ase import Atoms
        
        # If ase_atoms already exists in data (from pacemaker scraper), use it directly
        if 'ase_atoms' in data:
            return data['ase_atoms']
                
        # Extract positions, types, and lattice from FitSNAP data
        positions = data['Positions']
        atom_types = data['AtomTypes'] 
        # Check for both possible lattice keys (QMLattice from pacemaker, Lattice from other scrapers)
        lattice = data.get('Lattice', data.get('QMLattice'))
                
        # Handle atom types - can be either numeric indices or chemical symbols
        pyace_config = self.config.sections["PYACE"]
        elements = pyace_config.elements
        
        symbols = []
        for atom_type in atom_types:
            if isinstance(atom_type, str) and not atom_type.isdigit():
                # atom_type is already a chemical symbol (e.g., 'Ta', 'H')
                if atom_type in elements:
                    symbols.append(atom_type)
                else:
                    # Fallback - use first element if symbol not in elements list
                    self.pt.single_print(f"WARNING: atom symbol {atom_type} not in elements list {elements}, using {elements[0]}")
                    symbols.append(elements[0])
            else:
                # atom_type is numeric (LAMMPS style - 1-indexed integers)
                try:
                    element_idx = int(atom_type) - 1
                    if 0 <= element_idx < len(elements):
                        symbols.append(elements[element_idx])
                    else:
                        # Fallback - use first element
                        self.pt.single_print(f"WARNING: atom type {atom_type} out of range, using {elements[0]}")
                        symbols.append(elements[0])
                except ValueError:
                    # Shouldn't happen, but just in case
                    self.pt.single_print(f"WARNING: could not parse atom type {atom_type}, using {elements[0]}")
                    symbols.append(elements[0])
                
        # Create ASE Atoms object
        ase_atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=lattice,
            pbc=True
        )
        
        return ase_atoms
    
    def _get_atomic_environment(self, ase_atoms):
        """
        Convert ASE atoms to PyACE atomic environment
        """
        
        # Get cutoff from basis
        cutoff = self.ace_basis.cutoffmax
        
        # Create elements mapper dict
        elements_mapper_dict = {el: i for i, el in enumerate(self.ace_basis.elements_name)}
        
        # Create atomic environment
        try:
            atomic_env = aseatoms_to_atomicenvironment(
                ase_atoms, 
                cutoff=cutoff,
                elements_mapper_dict=elements_mapper_dict
            )
            return atomic_env
        except Exception as e:
            self.pt.single_print(f"ERROR creating atomic environment: {e}")
            raise e
    
    def _compute_ace_descriptors(self, atomic_env):
        """
        Compute ACE descriptors using PyACE evaluator
        """
        
        # Create ACECalculator and set evaluator
        ace_calc = ACECalculator()
        if hasattr(self.ace_basis, 'evaluator'):
            ace_calc.set_evaluator(self.ace_basis.evaluator)
        else:
            # Create evaluator from basis
            from pyace.evaluator import ACEBEvaluator
            evaluator = ACEBEvaluator(self.ace_basis)
            ace_calc.set_evaluator(evaluator)
        
        # Compute descriptors and derivatives
        ace_calc.compute(atomic_env, compute_projections=True, compute_b_grad=True)
        
        # Extract results
        energy = ace_calc.energy
        forces = np.array(ace_calc.forces)
        projections = np.array(ace_calc.projections)
        force_descriptors = np.array(ace_calc.forces_bfuncs)
                
        # Reshape projections to per-atom descriptors
        n_atoms = atomic_env.n_atoms_real
        if len(projections.shape) == 1:
            # Reshape to (n_atoms, n_descriptors)
            n_descriptors = len(projections) // n_atoms
            descriptors = projections.reshape((n_atoms, n_descriptors))
        else:
            descriptors = projections
        
        return {
            'energy': energy,
            'forces': forces,
            'descriptors': descriptors,
            'force_descriptors': force_descriptors,
            'ref_energy': 0.0,  # PyACE doesn't have reference potential
            'ref_forces': np.zeros_like(forces)
        }
    
    def _store_fitsnap_results(self, ace_results, data):
        """
        Store PyACE results in FitSNAP shared arrays format
        """
        
        # Calculate number of atoms from existing data (pacemaker scraper doesn't set NumAtoms)
        num_atoms = len(data['Positions'])
        energy = data['Energy']
        forces = data.get('Forces', np.zeros((num_atoms, 3)))
        
        # Get ACE results
        ace_energy = ace_results['energy']
        ace_forces = ace_results['forces']
        descriptors = ace_results['descriptors']
        force_descriptors = ace_results['force_descriptors']
        ref_energy = ace_results.get('ref_energy', 0.0)
        ref_forces = ace_results.get('ref_forces', np.zeros_like(ace_forces))
                
        # Get current indices
        index = self.shared_index
        dindex = self.distributed_index
                
        # Store energy descriptors
        if self.config.sections["CALCULATOR"].energy:
            # Sum descriptors over atoms for energy (following SNAP pattern)
            energy_descriptors = np.sum(descriptors, axis=0) / num_atoms
            self.pt.shared_arrays['a'].array[index] = energy_descriptors
            
            # Store energy truth (subtract reference energy)
            self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            
            # Store energy weight
            eweight = data.get('eweight', 1.0)
            self.pt.shared_arrays['w'].array[index] = eweight
            
            # Update fitsnap dictionaries
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + 1] = ['Energy']
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + 1] = [0]
            
            index += 1
            dindex += 1
                    
        # Store force descriptors  
        if self.config.sections["CALCULATOR"].force:
            # For forces, we need descriptor derivatives
            # This is a simplified approach - in reality we'd need dB/dr
            nrows_force = 3 * num_atoms
            
            force_descriptors = force_descriptors.transpose(2, 0, 1).reshape(nrows_force, -1)
            self.pt.shared_arrays['a'].array[index:index+nrows_force] = force_descriptors
            self.pt.single_print(f"*** force_descriptors {force_descriptors.shape} {force_descriptors}")
            
            # Store force truths (subtract reference forces)
            force_truth = (self._data["Forces"] - ref_forces).ravel()
            self.pt.shared_arrays['b'].array[index:index+nrows_force] = force_truth
            
            # Store force weights
            fweight = data.get('fweight', 1.0)
            self.pt.shared_arrays['w'].array[index:index+nrows_force] = fweight
            
            # Update fitsnap dictionaries
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex+nrows_force] = ['Force'] * nrows_force
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex+nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            
            # Set atom types for force rows
            atom_types = data['AtomTypes']
            force_atom_types = []
            for atom_type in atom_types:
                if isinstance(atom_type, str) and not atom_type.isdigit():
                    # atom_type is a chemical symbol, convert to numeric index
                    pyace_config = self.config.sections["PYACE"]
                    elements = pyace_config.elements
                    if atom_type in elements:
                        numeric_type = elements.index(atom_type) + 1  # 1-indexed
                    else:
                        numeric_type = 1  # Fallback to first type
                else:
                    # atom_type is already numeric
                    try:
                        numeric_type = int(atom_type)
                    except ValueError:
                        numeric_type = 1  # Fallback
                
                for _ in range(3):  # 3 force components per atom
                    force_atom_types.append(numeric_type)
            self.pt.fitsnap_dict['Atom_Type'][dindex:dindex+nrows_force] = force_atom_types
            
            index += nrows_force
            dindex += nrows_force
                    
        # Store stress descriptors
        if self.config.sections["CALCULATOR"].stress:
            # Placeholder for stress - would need virial derivatives
            nrows_stress = 6
            stress_descriptors = np.tile(np.sum(descriptors, axis=0), (nrows_stress, 1))
            
            self.pt.shared_arrays['a'].array[index:index+nrows_stress] = stress_descriptors
            
            # Store stress truths
            stress_data = data.get('Stress', np.zeros(6))
            self.pt.shared_arrays['b'].array[index:index+nrows_stress] = stress_data
            
            # Store stress weights
            vweight = data.get('vweight', 1.0)
            self.pt.shared_arrays['w'].array[index:index+nrows_stress] = vweight
            
            # Update fitsnap dictionaries
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex+nrows_stress] = ['Stress'] * nrows_stress
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex+nrows_stress] = [0] * nrows_stress
            
            index += nrows_stress
            dindex += nrows_stress
                    
        # Update group and config information
        length = dindex - self.distributed_index
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = [data['Group']] * length
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = [data['File']] * length
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(data.get('test_bool', False))] * length
        
        # Update indices
        self.shared_index = index
        self.distributed_index = dindex
        
        self.pt.single_print(self.pt.shared_arrays['a'].array[:3])
        self.pt.single_print(self.pt.shared_arrays['b'].array[:10])

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
        
        self.pt.single_print(f"PyACE config: elements={ace_config_dict['elements']}, cutoff={ace_config_dict['cutoff']}")
        
        try:
            # Create pyace basis from configuration using the imported pyace classes
            self.ace_basis = self._create_basis_from_config(ace_config_dict)
            
            # Create pyace calculator
            if self.ace_basis is not None:
                self.pyace_calc = PyACECalculator(self.ace_basis)
                width = self.get_width()
                self.pt.single_print(f"Successfully created pyace calculator with {width} basis functions")
            else:
                self.pt.single_print("Warning: Could not create pyace basis, calculator not initialized")
                
        except Exception as e:
            self.pt.single_print(f"Warning: Error setting up pyace calculator: {e}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            self.ace_basis = None
            self.pyace_calc = None
    
    def _create_basis_from_config(self, config_dict):
        """Create pyace basis from configuration dictionary"""
        try:
            self.pt.single_print(f"Creating BBasisConfiguration using create_multispecies_basis_config")
            self.pt.single_print(f"Config: elements={config_dict.get('elements')}, cutoff={config_dict.get('cutoff')}")
            
            # Use the proper PyACE function to create BBasisConfiguration
            basis_config = create_multispecies_basis_config(config_dict)
            
            # Apply lmin trimming if constraints are specified
            pyace_config = self.config.sections["PYACE"]
            if hasattr(pyace_config, 'lmin_constraints'):
                self.pt.single_print(f"Applying lmin constraints: {pyace_config.lmin_constraints}")
                from fitsnap3lib.io.sections.calculator_sections.pyace import PyAce
                basis_config = PyAce.trim_basis_configuration_for_lmin(basis_config, pyace_config.lmin_constraints)
            
            # Create ACEBBasisSet using the BBasisConfiguration
            ace_basis = ACEBBasisSet(basis_config)
            self.pt.single_print("Successfully created ACEBBasisSet using proper PyACE API")
            return ace_basis
                    
        except Exception as e:
            self.pt.single_print(f"Error creating pyace basis from config: {e}")
            self.pt.single_print(f"Invoked with: config_dict: {config_dict}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
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
        

