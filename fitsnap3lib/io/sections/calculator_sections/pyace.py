import numpy as np
import json
import itertools
from fitsnap3lib.io.sections.sections import Section


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
            'function_coefs_init',
            
            # Backwards compatibility with ACE section (flattened lists)
            'numTypes', 'type', 'bzeroflag', 'ranks', 'lmin', 'lmax', 'nmax',
            'nmaxbase', 'rcutfac', 'lambda', 'rcinner', 'drcinner'
        ]
        
        # Check for unknown keys
        for value_name in config['PYACE']:
            if value_name in allowedkeys: 
                continue
            else:
                raise RuntimeError(f">>> Found unmatched variable in PYACE section of input: {value_name}")
        
        # Parse configuration
        self._parse_basic_settings(config)
        self._parse_embeddings(config)
        self._parse_bonds(config) 
        self._parse_functions(config)
        self._setup_type_mapping()
        
        # Store for later use by calculator
        self.ace_config = self._build_ace_config()
        
    def _parse_basic_settings(self, config):
        """Parse basic PYACE settings"""
        # Elements list - either from 'elements' or legacy 'type'
        elements_str = self.get_value("PYACE", "elements", 
                                     self.get_value("PYACE", "type", "H"))
        self.elements = elements_str.split()
        self.numtypes = len(self.elements)
        
        # Global cutoff
        self.cutoff = self.get_value("PYACE", "cutoff", "10.0", "float")
        
        # Delta spline bins
        self.delta_spline_bins = self.get_value("PYACE", "delta_spline_bins", "0.001", "float")
        
        # Legacy compatibility
        self.bzeroflag = self.get_value("PYACE", "bzeroflag", "0", "bool")
        
    def _parse_embeddings(self, config):
        """Parse embedding configuration"""
        # Try JSON format first
        embeddings_str = self.get_value("PYACE", "embeddings", "")
        
        if embeddings_str:
            try:
                self.embeddings = json.loads(embeddings_str)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Error parsing embeddings: {e}")
        else:
            # Use simple/default embedding for all elements
            npot = self.get_value("PYACE", "embedding_npot", "FinnisSinclairShiftedScaled")
            fs_params_str = self.get_value("PYACE", "embedding_fs_parameters", "[1, 1]")
            fs_parameters = json.loads(fs_params_str)
            ndensity = self.get_value("PYACE", "embedding_ndensity", "1", "int")
            rho_core_cut = self.get_value("PYACE", "embedding_rho_core_cut", "200000", "float")
            drho_core_cut = self.get_value("PYACE", "embedding_drho_core_cut", "250", "float")
            
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
        bonds_str = self.get_value("PYACE", "bonds", "")
        
        if bonds_str:
            try:
                self.bonds = json.loads(bonds_str)
            except json.JSONDecodeError as e:
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
                
                # Map to bond pairs
                self.bonds = {}
                for i, bond_name in enumerate(self.bond_pair_names):
                    if i < len(rcutfac_vals):
                        self.bonds[bond_name] = {
                            'radbase': 'ChebExpCos',
                            'radparameters': [lambda_vals[i]],
                            'rcut': rcutfac_vals[i],
                            'dcut': 0.01,
                            'r_in': rcinner_vals[i],
                            'delta_in': drcinner_vals[i],
                            'core-repulsion': [100.0, 5.0]
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
                
                self.bonds = {
                    'ALL': {
                        'radbase': radbase,
                        'radparameters': radparameters,
                        'rcut': rcut,
                        'dcut': dcut,
                        'r_in': r_in,
                        'delta_in': delta_in,
                        'core-repulsion': core_repulsion
                    }
                }
            
    def _parse_functions(self, config):
        """Parse function configuration"""
        # Try JSON format first
        functions_str = self.get_value("PYACE", "functions", "")
        
        if functions_str:
            try:
                self.functions = json.loads(functions_str)
            except json.JSONDecodeError as e:
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
                self.functions = {
                    'UNARY': {
                        'nradmax_by_orders': nmax_vals,
                        'lmax_by_orders': lmax_vals,
                        'coefs_init': 'zero'
                    },
                    'BINARY': {
                        'nradmax_by_orders': nmax_vals,
                        'lmax_by_orders': lmax_vals,
                        'coefs_init': 'zero'
                    }
                }
            else:
                # Use simple/default functions
                nradmax_str = self.get_value("PYACE", "function_nradmax_by_orders", "[15, 3, 2, 2, 1]")
                lmax_str = self.get_value("PYACE", "function_lmax_by_orders", "[0, 2, 2, 1, 1]")
                nradmax_by_orders = json.loads(nradmax_str)
                lmax_by_orders = json.loads(lmax_str)
                coefs_init = self.get_value("PYACE", "function_coefs_init", "zero")
                
                self.functions = {
                    'UNARY': {
                        'nradmax_by_orders': nradmax_by_orders,
                        'lmax_by_orders': lmax_by_orders,
                        'coefs_init': coefs_init
                    },
                    'BINARY': {
                        'nradmax_by_orders': nradmax_by_orders[:-1],  # One less order for binary
                        'lmax_by_orders': lmax_by_orders[:-1],
                        'coefs_init': coefs_init
                    }
                }
            
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
        ace_config = {
            'cutoff': self.cutoff,
            'deltaSplineBins': self.delta_spline_bins,
            'elements': self.elements,
            'embeddings': self.embeddings,
            'bonds': self.bonds,
            'functions': self.functions
        }
        return ace_config
        
    def get_width(self):
        """
        Get width of descriptor vector for PYACE calculator.
        This is a placeholder - actual width calculation is done in the calculator.
        """
        # Return None to indicate calculator should determine width
        return None
        
    def get_ace_config(self):
        """Return the ACE configuration for use by calculator"""
        return self.ace_config
