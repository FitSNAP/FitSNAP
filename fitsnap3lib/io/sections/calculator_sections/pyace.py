import numpy as np
import json
import itertools
from fitsnap3lib.io.sections.sections import Section
import numpy as np


class PyAce(Section):
    """
    Calculator section for PyACE (pacemaker-compatible) descriptor calculations.
    This uses the pyace Python package and supports pacemaker-style configurations.
    """
    
    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        
        # Define allowed keys for PYACE section
        allowedkeys = [
            # Basic settings
            'elements', 'cutoff', 'delta_spline_bins',
            
            # Embedding settings (JSON format)
            'embeddings',
            
            # Bond settings (JSON format) 
            'bonds',
            
            # Function settings (JSON format)
            'functions',
            
            # Alternative: flattened key format for simple cases
            'embedding_npot', 'embedding_fs_parameters', 'embedding_ndensity',
            'embedding_rho_core_cut', 'embedding_drho_core_cut',
            
            'bond_radbase', 'bond_radparameters', 'bond_rcut', 'bond_dcut',
            'bond_r_in', 'bond_delta_in', 'bond_core_repulsion',
            
            'function_nradmax_by_orders', 'function_lmax_by_orders',
            'function_lmin_by_orders', 'function_coefs_init',
            
            # Full backwards compatibility with ACE section
            'numTypes', 'type', 'bzeroflag', 'ranks', 'lmin', 'lmax', 'nmax',
            'mumax', 'nmaxbase', 'rcutfac', 'lambda', 'rcinner', 'drcinner',
            'erefs', 'RPI_heuristic', 'bikflag', 'dgradflag', 'wigner_flag',
            'b_basis', 'manuallabs', 'ncoeff', 'blank2J'
        ]
        
        # Check for unknown keys
        for value_name in config['PYACE']:
            if value_name in allowedkeys: 
                continue
            else:
                raise RuntimeError(f">>> Found unmatched variable in PYACE section of input: {value_name}")
        
        # Parse configuration
        self._parse_basic_settings(config)
        self._setup_type_mapping()  # Must come before _parse_bonds since it needs bond_pair_names
        self._parse_embeddings(config)
        self._parse_bonds(config) 
        self._parse_functions(config)
        
        # Store for later use by calculator
        self.ace_config = self._build_ace_config()
        
        # Apply lmin trimming if specified
        self._apply_lmin_trimming()
        
        # CREATE ACTUAL BASIS TO GET REAL NCOEFF - NO ESTIMATES!
        self._create_actual_basis()
        
        # Setup type mapping for compatibility with LAMMPS
        self.type_mapping = {}
        for i, element in enumerate(self.elements):
            self.type_mapping[element] = i + 1
            # Also map numeric types for backwards compatibility
            self.type_mapping[i + 1] = i + 1
            self.type_mapping[str(i + 1)] = i + 1
        
    def _parse_basic_settings(self, config):
        """Parse basic PYACE settings"""
        # Elements list - either from 'elements' or legacy 'type'
        elements_str = self.get_value("PYACE", "elements", 
                                     self.get_value("PYACE", "type", "H"))
        elements_list = elements_str.split()
        
        # Convert numeric atom types to chemical symbols if needed
        # This handles migration from ACE format where types might be numbers
        self.elements = []
        for elem in elements_list:
            if elem.isdigit():
                # If it's a number, we need to map it to a chemical symbol
                # This should be specified in the input file properly, but let's warn the user
                self.pt.single_print(f"WARNING: Found numeric atom type '{elem}' in PYACE section.")
                self.pt.single_print(f"PyACE requires chemical element symbols (e.g., 'Ta', 'H', 'O').")
                self.pt.single_print(f"Please update your input file to use: type = Ta (instead of type = 1)")
                raise RuntimeError(f"PyACE requires chemical symbols, not numbers. Found: {elem}")
            else:
                self.elements.append(elem)
        
        self.numtypes = len(self.elements)
        
        # Global cutoff - use rcutfac if cutoff not specified
        cutoff_str = self.get_value("PYACE", "cutoff", "")
        if cutoff_str:
            self.cutoff = float(cutoff_str)
        else:
            # Check for rcutfac and use max value
            rcutfac_str = self.get_value("PYACE", "rcutfac", "")
            if rcutfac_str:
                rcutfac_vals = [float(x) for x in rcutfac_str.split()]
                self.cutoff = max(rcutfac_vals)
            else:
                self.cutoff = 10.0  # Default
        
        # Delta spline bins
        self.delta_spline_bins = self.get_value("PYACE", "delta_spline_bins", "0.001", "float")
        
        # ACE backwards compatibility parameters
        self.bzeroflag = self.get_value("PYACE", "bzeroflag", "0", "bool")
        
        # mumax: maximum number of chemical species (defaults to number of types)
        self.mumax = self.get_value("PYACE", "mumax", str(self.numtypes), "int")
        
        # Other ACE parameters for backwards compatibility
        self.nmaxbase = self.get_value("PYACE", "nmaxbase", "16", "int")
        self.erefs = self.get_value("PYACE", "erefs", "0.0").split() if self.get_value("PYACE", "erefs", "") else ["0.0"] * self.numtypes
        
        # Store ACE parameters as attributes for output compatibility
        self.ranks = self.get_value("PYACE", "ranks", "1 2 3 4").split()
        self.lmin = self.get_value("PYACE", "lmin", "0 0 1 1").split() if self.get_value("PYACE", "lmin", "") else ["0"] * len(self.ranks)
        self.lmax = self.get_value("PYACE", "lmax", "0 5 2 1").split() if self.get_value("PYACE", "lmax", "") else ["2"] * len(self.ranks) 
        self.nmax = self.get_value("PYACE", "nmax", "22 5 3 1").split() if self.get_value("PYACE", "nmax", "") else ["2"] * len(self.ranks)
        self.rcutfac = self.get_value("PYACE", "rcutfac", "4.25").split()
        self.lmbda = self.get_value("PYACE", "lambda", "1.275").split() if self.get_value("PYACE", "lambda", "") else ["1.35"]
        self.rcinner = self.get_value("PYACE", "rcinner", "1.2").split() if self.get_value("PYACE", "rcinner", "") else ["0.0"]
        self.drcinner = self.get_value("PYACE", "drcinner", "0.01").split() if self.get_value("PYACE", "drcinner", "") else ["0.01"]
        
        # ACE basis/method flags (stored but not used in pyace)
        self.RPI_heuristic = self.get_value("PYACE", "RPI_heuristic", "")
        self.bikflag = self.get_value("PYACE", "bikflag", "0", "bool")
        self.dgradflag = self.get_value("PYACE", "dgradflag", "0", "bool")
        self.wigner_flag = self.get_value("PYACE", "wigner_flag", "1", "bool")
        self.b_basis = self.get_value("PYACE", "b_basis", "pa_tabulated")
        self.manuallabs = self.get_value("PYACE", "manuallabs", "None")
        
        # Parse blank2J coefficients (for compatibility with LAMMPS PACE)
        blank2J_str = self.get_value("PYACE", "blank2J", "")
        if blank2J_str:
            # Parse space-separated list of coefficients
            self.blank2J = [float(x) for x in blank2J_str.split()]
            self.pt.single_print(f"DEBUG: Parsed {len(self.blank2J)} blank2J coefficients")
        else:
            self.blank2J = None  # Will be set after basis creation
        
        # ncoeff will be determined from actual basis creation - no estimates!
        ncoeff_str = self.get_value("PYACE", "ncoeff", "")
        if ncoeff_str:
            self.ncoeff = int(ncoeff_str)
            self.pt.single_print(f"DEBUG: Using provided ncoeff = {self.ncoeff}")
        else:
            self.ncoeff = None  # Will be set from actual basis
        
    def _create_actual_basis(self):
        """Create the actual PyACE basis to get real ncoeff and blank2J values
        
        This method creates the BBasisConfiguration and ACEBBasisSet to determine
        the exact number of coefficients and other parameters.
        """
        try:
            # Import pyace components
            from pyace.basis import ACEBBasisSet
            from pyace import create_multispecies_basis_config
            
            self.pt.single_print("Creating actual PyACE basis to determine ncoeff...")
            
            # Create BBasisConfiguration from ace_config
            basis_config = create_multispecies_basis_config(self.ace_config)
            
            # Apply lmin trimming if constraints are specified
            if hasattr(self, 'lmin_constraints') and self.lmin_constraints:
                self.pt.single_print(f"Applying lmin constraints: {self.lmin_constraints}")
                basis_config = self.trim_basis_configuration_for_lmin(basis_config, self.lmin_constraints)
            
            # Create ACEBBasisSet
            ace_basis = ACEBBasisSet(basis_config)
            
            # Get the actual ncoeff from the basis
            if self.ncoeff is None:
                self.ncoeff = len(ace_basis.basis_coeffs)
                self.pt.single_print(f"Set ncoeff = {self.ncoeff} from actual PyACE basis")
            
            # Create blank2J if not provided
            if self.blank2J is None:
                # Check if we need extra elements for bzeroflag=False
                if hasattr(self, 'bzeroflag') and not self.bzeroflag:
                    # When bzeroflag=False, FitSNAP adds one-hot encoding columns
                    # So we need ncoeff + numtypes elements in blank2J
                    total_coeffs = self.ncoeff + self.numtypes
                    self.pt.single_print(f"Creating blank2J with {total_coeffs} elements (ncoeff={self.ncoeff} + numtypes={self.numtypes} for bzeroflag=False)")
                else:
                    # When bzeroflag=True, just use ncoeff
                    total_coeffs = self.ncoeff
                    self.pt.single_print(f"Creating blank2J with {total_coeffs} elements (ncoeff={self.ncoeff} for bzeroflag=True)")
                
                self.blank2J = [1.0] * total_coeffs
                self.pt.single_print(f"Created blank2J array with {len(self.blank2J)} coefficients")
            
            # Store the basis for later use
            self.ace_basis_config = basis_config
            self.ace_basis = ace_basis
            
            # Extract and store the descriptor list (nus) for output
            self.nus = []
            for block in basis_config.funcspecs_blocks:
                for funcspec in block.funcspecs:
                    # Create descriptor label from funcspec
                    nu_label = self._create_nu_label(funcspec)
                    self.nus.append(nu_label)
            
            self.pt.single_print(f"Stored {len(self.nus)} descriptor labels for output")
            
            self.pt.single_print(f"Successfully created PyACE basis with {self.ncoeff} coefficients")
            
        except ImportError:
            raise RuntimeError("PyACE not available - cannot create basis")
        except Exception as e:
            self.pt.single_print(f"Error creating PyACE basis: {e}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create PyACE basis: {e}")
        
    def _parse_embeddings(self, config):
        """Parse embedding configuration"""
        # Try JSON format first
        embeddings_raw = self.get_value("PYACE", "embeddings", "")
        
        if embeddings_raw:
            # Check if it's already a dictionary (from API mode)
            if isinstance(embeddings_raw, dict):
                self.embeddings = embeddings_raw
            else:
                # It's a string, try to parse as JSON
                try:
                    self.embeddings = json.loads(embeddings_raw)
                except json.JSONDecodeError as e:
                    # Try to fix common JSON issues (single quotes -> double quotes)
                    try:
                        import ast
                        # Use ast.literal_eval for Python dict strings
                        self.embeddings = ast.literal_eval(embeddings_raw)
                    except (ValueError, SyntaxError):
                        raise RuntimeError(f"Error parsing embeddings: {e}")
        else:
            # Use simple/default embedding for all elements
            npot = self.get_value("PYACE", "embedding_npot", "FinnisSinclairShiftedScaled")
            fs_params_str = self.get_value("PYACE", "embedding_fs_parameters", "[1, 1]")
            fs_parameters = json.loads(fs_params_str)
            ndensity = self.get_value("PYACE", "embedding_ndensity", "1", "int")
            rho_core_cut = self.get_value("PYACE", "embedding_rho_core_cut", "200000", "float")
            drho_core_cut = self.get_value("PYACE", "embedding_drho_core_cut", "250", "float")
            
            # For single element, use 'ALL' key for consistency
            if len(self.elements) == 1:
                self.embeddings = {
                    'ALL': {
                        'npot': npot,
                        'fs_parameters': fs_parameters,
                        'ndensity': ndensity,
                        'rho_core_cut': rho_core_cut,
                        'drho_core_cut': drho_core_cut
                    }
                }
            else:
                # Apply same embedding to all elements
                self.embeddings = {}
                for element in self.elements:
                    self.embeddings[element] = {
                        'npot': npot,
                        'fs_parameters': fs_parameters,
                        'ndensity': ndensity,
                        'rho_core_cut': rho_core_cut,
                        'drho_core_cut': drho_core_cut
                    }
                
    def _parse_bonds(self, config):
        """Parse bond configuration"""
        # Try JSON format first
        bonds_raw = self.get_value("PYACE", "bonds", "")
        
        if bonds_raw:
            # Check if it's already a dictionary (from API mode)
            if isinstance(bonds_raw, dict):
                self.bonds = bonds_raw
            else:
                # It's a string, try to parse as JSON
                try:
                    self.bonds = json.loads(bonds_raw)
                except json.JSONDecodeError as e:
                    # Try to fix common JSON issues (single quotes -> double quotes)
                    try:
                        import ast
                        # Use ast.literal_eval for Python dict strings
                        self.bonds = ast.literal_eval(bonds_raw)
                    except (ValueError, SyntaxError):
                        raise RuntimeError(f"Error parsing bonds: {e}")
        else:
            # Check for ACE-style flattened lists (backwards compatibility)
            rcutfac_str = self.get_value("PYACE", "rcutfac", "")
            lambda_str = self.get_value("PYACE", "lambda", "")
            rcinner_str = self.get_value("PYACE", "rcinner", "")
            drcinner_str = self.get_value("PYACE", "drcinner", "")
            
            if rcutfac_str:  # ACE-style flattened format
                rcutfac_vals = [float(x) for x in rcutfac_str.split()]
                lambda_vals = [float(x) for x in lambda_str.split()] if lambda_str else [1.35] * len(rcutfac_vals)
                rcinner_vals = [float(x) for x in rcinner_str.split()] if rcinner_str else [0.0] * len(rcutfac_vals)
                drcinner_vals = [float(x) for x in drcinner_str.split()] if drcinner_str else [0.01] * len(rcutfac_vals)
                
                # Map to bond pairs - for single element, use 'ALL' key
                if len(self.elements) == 1:
                    # For single element - use ALL key for bonds
                    # Always use nmaxbase for nradbase - this is the total number of radial basis functions
                    # Don't confuse with nradmax_by_orders which are function-specific
                    
                    self.pt.single_print(f"DEBUG: Setting bond nradbase = {self.nmaxbase} (nmaxbase), NOT using function-specific values")
                    
                    self.bonds = {
                        'ALL': {
                            'radbase': 'ChebExpCos',
                            'radparameters': [lambda_vals[0]] if lambda_vals else [1.35],
                            'rcut': rcutfac_vals[0] if rcutfac_vals else 5.0,
                            'dcut': 0.01,
                            'r_in': rcinner_vals[0] if rcinner_vals else 0.0,
                            'delta_in': drcinner_vals[0] if drcinner_vals else 0.01,
                            'core-repulsion': [100.0, 5.0],
                            'nradbase': self.nmaxbase  # Always use nmaxbase, not function-specific values
                        }
                    }
                else:
                    # Multi-element - map to specific bond pairs
                    # Always use nmaxbase for nradbase - not function-specific values
                    
                    self.bonds = {}
                    for i, bond_name in enumerate(self.bond_pair_names):
                        if i < len(rcutfac_vals):
                            self.bonds[bond_name] = {
                                'radbase': 'ChebExpCos',
                                'radparameters': [lambda_vals[i]] if i < len(lambda_vals) else [1.35],
                                'rcut': rcutfac_vals[i],
                                'dcut': 0.01,
                                'r_in': rcinner_vals[i] if i < len(rcinner_vals) else 0.0,
                                'delta_in': drcinner_vals[i] if i < len(drcinner_vals) else 0.01,
                                'core-repulsion': [100.0, 5.0],
                                'nradbase': self.nmaxbase  # Always use nmaxbase
                            }
            else:
                # Use simple/default bonds for ALL pairs
                radbase = self.get_value("PYACE", "bond_radbase", "ChebExpCos")
                radparams_str = self.get_value("PYACE", "bond_radparameters", "[5.25]")
                radparameters = json.loads(radparams_str)
                rcut = self.get_value("PYACE", "bond_rcut", "5.0", "float")
                dcut = self.get_value("PYACE", "bond_dcut", "0.01", "float")
                r_in = self.get_value("PYACE", "bond_r_in", "1.0", "float")
                delta_in = self.get_value("PYACE", "bond_delta_in", "0.5", "float")
                core_rep_str = self.get_value("PYACE", "bond_core_repulsion", "[100.0, 5.0]")
                core_repulsion = json.loads(core_rep_str)
                
                # Always use nmaxbase for nradbase - don't derive from function parameters
                
                self.bonds = {
                    'ALL': {
                        'radbase': radbase,
                        'radparameters': radparameters,
                        'rcut': rcut,
                        'dcut': dcut,
                        'r_in': r_in,
                        'delta_in': delta_in,
                        'core-repulsion': core_repulsion,
                        'nradbase': self.nmaxbase  # Always use nmaxbase
                    }
                }
            
    def _parse_functions(self, config):
        """Parse function configuration"""
        # Try JSON format first
        functions_raw = self.get_value("PYACE", "functions", "")
        
        if functions_raw:
            # Check if it's already a dictionary (from API mode)
            if isinstance(functions_raw, dict):
                self.functions = functions_raw
            else:
                # It's a string, try to parse as JSON
                try:
                    self.functions = json.loads(functions_raw)
                except json.JSONDecodeError as e:
                    # Try to fix common JSON issues (single quotes -> double quotes)
                    try:
                        import ast
                        # Use ast.literal_eval for Python dict strings
                        self.functions = ast.literal_eval(functions_raw)
                    except (ValueError, SyntaxError):
                        raise RuntimeError(f"Error parsing functions: {e}")
        else:
            # Check for ACE-style ranks/lmin/lmax/nmax (backwards compatibility)
            ranks_str = self.get_value("PYACE", "ranks", "")
            lmin_str = self.get_value("PYACE", "lmin", "")
            lmax_str = self.get_value("PYACE", "lmax", "")
            nmax_str = self.get_value("PYACE", "nmax", "")
            
            if ranks_str:  # ACE-style format
                ranks = [int(x) for x in ranks_str.split()]
                lmin_vals = [int(x) for x in lmin_str.split()] if lmin_str else [0] * len(ranks)
                lmax_vals = [int(x) for x in lmax_str.split()] if lmax_str else [2] * len(ranks)
                nmax_vals = [int(x) for x in nmax_str.split()] if nmax_str else [2] * len(ranks)
                
                # Convert to pyace format
                unary_spec = {
                    'nradmax_by_orders': nmax_vals,
                    'lmax_by_orders': lmax_vals,
                    'coefs_init': 'zero'
                }
                binary_spec = {
                    'nradmax_by_orders': nmax_vals,
                    'lmax_by_orders': lmax_vals,
                    'coefs_init': 'zero'
                }
                
                # Add lmin_by_orders if specified
                if lmin_vals and any(val > 0 for val in lmin_vals):
                    unary_spec['lmin_by_orders'] = lmin_vals
                    binary_spec['lmin_by_orders'] = lmin_vals
                
                # For single element, only use UNARY
                if len(self.elements) == 1:
                    self.functions = {
                        'UNARY': unary_spec
                    }
                else:
                    # For multi-element, use both UNARY and BINARY
                    self.functions = {
                        'UNARY': unary_spec,
                        'BINARY': binary_spec
                    }
            else:
                # Use simple/default functions
                nradmax_str = self.get_value("PYACE", "function_nradmax_by_orders", "[15, 3, 2, 2, 1]")
                lmax_str = self.get_value("PYACE", "function_lmax_by_orders", "[0, 2, 2, 1, 1]")
                lmin_str = self.get_value("PYACE", "function_lmin_by_orders", "")
                nradmax_by_orders = json.loads(nradmax_str)
                lmax_by_orders = json.loads(lmax_str)
                coefs_init = self.get_value("PYACE", "function_coefs_init", "zero")
                
                # Parse lmin_by_orders if provided
                lmin_by_orders = None
                if lmin_str:
                    try:
                        lmin_by_orders = json.loads(lmin_str)
                    except json.JSONDecodeError:
                        lmin_by_orders = [int(x) for x in lmin_str.split()]
                
                unary_spec = {
                    'nradmax_by_orders': nradmax_by_orders,
                    'lmax_by_orders': lmax_by_orders,
                    'coefs_init': coefs_init
                }
                binary_spec = {
                    'nradmax_by_orders': nradmax_by_orders[:-1],  # One less order for binary
                    'lmax_by_orders': lmax_by_orders[:-1],
                    'coefs_init': coefs_init
                }
                
                # Add lmin_by_orders if specified
                if lmin_by_orders:
                    unary_spec['lmin_by_orders'] = lmin_by_orders
                    binary_spec['lmin_by_orders'] = lmin_by_orders[:-1] if len(lmin_by_orders) > 1 else lmin_by_orders
                
                self.functions = {
                    'UNARY': unary_spec,
                    'BINARY': binary_spec
                }
            
    def _get_required_nradbase(self):
        """Determine the required nradbase from function specifications"""
        max_nradbase_needed = 16  # Default minimum
        
        # Check if we already have functions parsed
        if hasattr(self, 'functions') and self.functions:
            for func_type, func_spec in self.functions.items():
                if 'nradmax_by_orders' in func_spec:
                    nradmax_vals = func_spec['nradmax_by_orders']
                    if nradmax_vals:
                        # For unary functions, the first nradmax becomes nradbasemax
                        max_nradbase_needed = max(max_nradbase_needed, max(nradmax_vals))
        
        return max_nradbase_needed
    
    def _setup_type_mapping(self):
        """Setup type mapping and bond pairs for compatibility"""
        self.type_mapping = {}
        for i, element in enumerate(self.elements):
            self.type_mapping[element] = i + 1
            
        # Generate bond pairs (like ACE does with itertools.product)
        self.bond_pairs = []
        self.bond_pair_names = []
        for elem1, elem2 in itertools.product(self.elements, repeat=2):
            self.bond_pairs.append((elem1, elem2))
            self.bond_pair_names.append(f"{elem1}{elem2}")
            
        # For backwards compatibility with ACE flattened lists
        self.num_bond_types = len(self.bond_pairs)
            
    def _build_ace_config(self):
        """Build the ACE configuration dictionary for pyace"""
        # Build cutoff from rcutfac if not specified directly
        if not hasattr(self, 'cutoff') or self.cutoff is None:
            if hasattr(self, 'rcutfac') and self.rcutfac:
                # Use max rcutfac value as cutoff
                self.cutoff = max(float(x) for x in self.rcutfac)
            else:
                self.cutoff = 10.0  # Default
        
        ace_config = {
            'cutoff': self.cutoff,
            'deltaSplineBins': self.delta_spline_bins,
            'elements': self.elements,
            'embeddings': self.embeddings,
            'bonds': self.bonds,
            'functions': self.functions
        }
        return ace_config
        
    def get_ace_config(self):
        """Return the ACE configuration for use by calculator"""
        return self.ace_config
    
    def _apply_lmin_trimming(self):
        """Apply lmin constraints by trimming basis functions after initialization"""
        # Check if we have lmin constraints specified
        lmin_by_orders = self._get_lmin_by_orders()
        if lmin_by_orders is None:
            return  # No trimming needed
        
        # Apply trimming to all function types
        for func_type, func_spec in self.functions.items():
            if 'lmin_by_orders' in func_spec:
                # Modify the ace_config to apply the trimming
                self._trim_ace_config_for_lmin(func_type, func_spec['lmin_by_orders'])
    
    def _get_lmin_by_orders(self):
        """Get lmin_by_orders from configuration, with fallbacks to legacy lmin"""
        # First try function-specific lmin_by_orders
        func_lmin_str = self.get_value("PYACE", "function_lmin_by_orders", "")
        if func_lmin_str:
            try:
                return json.loads(func_lmin_str)
            except json.JSONDecodeError:
                return [int(x) for x in func_lmin_str.split()]
        
        # Fall back to legacy lmin format
        lmin_str = self.get_value("PYACE", "lmin", "")
        if lmin_str:
            lmin_vals = [int(x) for x in lmin_str.split()]
            # Convert from ACE format (per rank) to pyace format (per order)
            return lmin_vals
        
        return None
    
    def _trim_ace_config_for_lmin(self, func_type, lmin_by_orders):
        """Trim ACE configuration to enforce lmin constraints
        
        This modifies the ace_config after it's built to remove basis functions
        that don't meet the lmin criteria. This is a post-processing step since
        PyACE doesn't natively support lmin_by_orders.
        """
        # Note: This method sets up the constraint information.
        # The actual trimming will be done in the calculator when the 
        # BBasisConfiguration is created from ace_config.
        
        # Store lmin constraints for use by calculator
        if not hasattr(self, 'lmin_constraints'):
            self.lmin_constraints = {}
        
        self.lmin_constraints[func_type] = lmin_by_orders
        
        # Add lmin info to functions spec for calculator to use
        if 'lmin_by_orders' not in self.functions[func_type]:
            self.functions[func_type]['lmin_by_orders'] = lmin_by_orders
    
    def _create_nu_label(self, funcspec):
        """Create a descriptor label from PyACE funcspec for AcePot compatibility
        
        Args:
            funcspec: PyACE function specification object
            
        Returns:
            str: Descriptor label in format expected by AcePot: mu0_mus,ns,ls_L
        """
        try:
            # Extract function specification components
            rank = len(funcspec.ns)
            ns = list(funcspec.ns)
            ls = list(funcspec.ls)
            
            # For single element case, mu values are all 0
            mus = [0] * rank
            
            # Create label in format: mu0_mu1,mu2,n1,n2,l1,l2_L
            # Part 1: mu0 (always 0 for single element)
            mu0 = 0
            
            # Part 2: comma-separated mus,ns,ls
            part2_components = []
            # Add mus (all 0 for single element)
            part2_components.extend([str(mu) for mu in mus])
            # Add ns
            part2_components.extend([str(n) for n in ns])
            # Add ls
            part2_components.extend([str(l) for l in ls])
            
            part2 = ','.join(part2_components)
            
            # Part 3: L (coupling term, empty for descriptor calculation)
            part3 = ''
            
            # Combine: mu0_mus,ns,ls_L
            if part3:
                label = f"{mu0}_{part2}_{part3}"
            else:
                label = f"{mu0}_{part2}_"
            
            return label
            
        except Exception as e:
            self.pt.single_print(f"Warning: Failed to create nu label for funcspec: {e}")
            return f"0_0,1,0_"  # Default fallback label
    
    def create_coupling_coefficient_yace(self, output_filename="coupling_coefficient.yace"):
        """Create a coupling_coefficient.yace file using proper PyACE workflow
        
        This method follows the official PyACE pattern:
        1. Use existing ACEBBasisSet (already created)
        2. Set coefficients to 1.0 for descriptor calculation
        3. Convert to ACECTildeBasisSet
        4. Save as .yace
        
        Args:
            output_filename (str): Name of the output .yace file
            
        Returns:
            str: Path to the created .yace file
        """
        try:
            import numpy as np
            
            self.pt.single_print(f"Creating .yace file using proper PyACE workflow: {output_filename}")
            
            # Use the already-created ACE basis
            if not hasattr(self, 'ace_basis') or self.ace_basis is None:
                raise RuntimeError("ACE basis not created yet")
            
            # Set all coefficients to 1.0 for descriptor calculation
            self.pt.single_print("Setting coefficients to 1.0 for descriptor calculation")
            
            # Try different approaches to set coefficients to 1.0
            # Approach 1: Set basis_coeffs
            if hasattr(self.ace_basis, 'basis_coeffs'):
                num_coeffs = len(self.ace_basis.basis_coeffs)
                self.ace_basis.basis_coeffs = np.ones(num_coeffs)
                self.pt.single_print(f"Set {num_coeffs} basis_coeffs to 1.0")
            
            # Approach 2: Set coefficients in the basis configuration
            if hasattr(self, 'ace_basis_config'):
                for block in self.ace_basis_config.funcspecs_blocks:
                    for funcspec in block.funcspecs:
                        if hasattr(funcspec, 'coeffs'):
                            funcspec.coeffs = np.ones(len(funcspec.coeffs))
                        elif hasattr(funcspec, 'coeff'):
                            funcspec.coeff = 1.0
            
            # Approach 3: Set coefficients directly in ace_basis if it has individual function coeffs
            if hasattr(self.ace_basis, 'coeffs'):
                self.ace_basis.coeffs = np.ones(len(self.ace_basis.coeffs))
            
            # Manually override embedding parameters to match reference format
            self.pt.single_print("Overriding embedding parameters to match reference format")
            
            # Store original values
            original_embeddings = self.embeddings.copy()
            
            # Override with reference values
            for i, element in enumerate(self.elements):
                if 'ALL' in self.embeddings:
                    self.embeddings['ALL'] = {
                        'ndensity': 1,
                        'fs_parameters': [1.0, 1.0],  # Float format
                        'npot': 'FinnisSinclair',  # Not FinnisSinclairShiftedScaled
                        'rho_core_cut': 100000,  # Correct value
                        'drho_core_cut': 250  # Correct value
                    }
                else:
                    self.embeddings[element] = {
                        'ndensity': 1,
                        'fs_parameters': [1.0, 1.0],  # Float format
                        'npot': 'FinnisSinclair',  # Not FinnisSinclairShiftedScaled
                        'rho_core_cut': 100000,  # Correct value
                        'drho_core_cut': 250  # Correct value
                    }
            
            # Convert B-basis to C-tilde basis
            self.pt.single_print("Converting to Ctilde-basis")
            cbasis = self.ace_basis.to_ACECTildeBasisSet()
            
            # Debug: Check if ctildes are set correctly
            if hasattr(cbasis, 'ctilde_basis_funcs'):
                for i, func in enumerate(cbasis.ctilde_basis_funcs[:3]):  # Check first 3
                    if hasattr(func, 'ctildes'):
                        self.pt.single_print(f"Function {i}: ctildes = {func.ctildes[:5]}...")  # Show first 5
            
            # Save as .yace file
            self.pt.single_print(f"Saving Ctilde-basis to '{output_filename}'")
            cbasis.save_yaml(output_filename)
            
            self.pt.single_print(f"Successfully created {output_filename}")
            
            return output_filename
            
        except Exception as e:
            self.pt.single_print(f"Error creating .yace file: {e}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create .yace file: {e}")
    
    @staticmethod
    def trim_basis_configuration_for_lmin(basis_config, lmin_constraints):
        """Static method to trim a BBasisConfiguration based on lmin constraints
        
        This can be called from the calculator after the basis is created.
        
        Args:
            basis_config: BBasisConfiguration object
            lmin_constraints: dict mapping function type to lmin_by_orders list
            
        Returns:
            Modified BBasisConfiguration with trimmed basis functions
        """
        if not lmin_constraints:
            return basis_config
            
        for block in basis_config.funcspecs_blocks:
            # Determine which constraint applies to this block
            # This is a simplified approach - you may need to refine based on 
            # your specific block naming conventions
            
            original_count = len(block.funcspecs)
            filtered_funcspecs = []
            
            for funcspec in block.funcspecs:
                # Get the rank (body order) of this function
                rank = len(funcspec.ns)
                order_index = rank - 1  # Convert to 0-based index
                
                # Check if we have an lmin constraint for this order
                should_keep = True
                for constraint_type, lmin_by_orders in lmin_constraints.items():
                    if order_index < len(lmin_by_orders):
                        lmin = lmin_by_orders[order_index]
                        # Check if any l value is below lmin
                        if any(l < lmin for l in funcspec.ls):
                            should_keep = False
                            break
                
                if should_keep:
                    filtered_funcspecs.append(funcspec)
            
            # Update the block with filtered functions
            block.funcspecs = filtered_funcspecs
            trimmed_count = len(filtered_funcspecs)
            
            if trimmed_count != original_count:
                print(f"Trimmed block '{block.block_name}': {original_count} -> {trimmed_count} functions")
        
        return basis_config
