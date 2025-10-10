import numpy as np
import json
import itertools
from fitsnap3lib.io.sections.sections import Section
import numpy as np


# ------------------------------------------------------------------------------------------------

class PyAce(Section):
    """
    Calculator section for PyACE (pacemaker-compatible) descriptor calculations.
    This uses the pyace Python package and supports pacemaker-style configurations.
    """
    
    # --------------------------------------------------------------------------------------------
    
    def __init__(self, name, config, pt, infile, args):
        super().__init__(name, config, pt, infile, args)
        
        # Define allowed keys for PYACE section
        allowedkeys = [
            # Basic settings
            'elements', 'cutoff', 'delta_spline_bins',
            
            # JSON format
            'embeddings', 'bonds', 'functions',
            
            # Full backwards compatibility with ACE section
            'numTypes', 'type', 'bzeroflag', 'ranks', 'lmin', 'lmax', 'nmax',
            'mumax', 'nmaxbase', 'rcutfac', 'lambda', 'rcinner', 'drcinner',
            'erefs', 'RPI_heuristic', 'bikflag', 'dgradflag', 'wigner_flag',
            'b_basis', 'manuallabs',
        ]
        
        # Check for unknown keys
        for value_name in config['PYACE']:
            if value_name in allowedkeys: 
                continue
            else:
                raise RuntimeError(f">>> Found unmatched variable in PYACE section of input: {value_name}")
        
        # Parse configuration
        self._parse_basic_settings(config)
        self._setup_type_mapping()  #
        self._parse_embeddings(config)
        self._parse_functions(config) # needs nradmax lmax nradbasemax before _parse_bonds()
        self._parse_bonds(config)
                
        # Store for later use by calculator
        self.ace_config = {
            'cutoff': self.cutoff,
            'deltaSplineBins': self.delta_spline_bins,
            'elements': self.elements,
            'embeddings': self.embeddings,
            'bonds': self.bonds,
            'functions': self.functions
        }
        
        self._create_basis()
        self._create_coupling_coefficients_yace()
      
    # --------------------------------------------------------------------------------------------

    def _parse_basic_settings(self, config):
        """Parse basic PYACE settings"""
        # Elements list - either from 'elements' or legacy 'type'
        elements_str = self.get_value("PYACE", "elements", self.get_value("PYACE", "type", "H"))
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
        self.types = self.elements
        
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
        self.bikflag = self.get_value("PYACE", "bikflag", "0", "bool")
        self.bzeroflag = self.get_value("PYACE", "bzeroflag", "0", "bool")
        self.dgradflag = self.get_value("PYACE", "dgradflag", "0", "bool")
                
        # Other ACE parameters for backwards compatibility
        self.erefs = self.get_value("PYACE", "erefs", "0.0").split() if self.get_value("PYACE", "erefs", "") else ["0.0"] * self.numtypes
        
    # --------------------------------------------------------------------------------------------
                        
    def _create_basis(self):
        """Create the actual PyACE basis to get real ncoeff value
        
        This method creates the BBasisConfiguration and ACEBBasisSet to determine
        the exact number of coefficients.
        """
        try:
            from pyace.basis import ACEBBasisSet, ACECTildeBasisSet
            from pyace import create_multispecies_basis_config
            
            basis_config = create_multispecies_basis_config(self.ace_config)
            b_basis = ACEBBasisSet(basis_config)
            b_basis.basis_coeffs = np.ones(len(b_basis.basis_coeffs))
            ctilde_basis = b_basis.to_ACECTildeBasisSet()
            self.ctilde_basis = ctilde_basis
                
            # total number of functions for all elements and ranks
            self.ncoeff = sum([len(f) for f in ctilde_basis.basis_rank1]) \
                        + sum([len(f) for f in ctilde_basis.basis])
                        
            self.pt.single_print(f"PyACE basis: numtypes {self.numtypes} ncoeff {self.ncoeff}")
            
            self.blist = [str(i) for i in range(self.ncoeff)]

            if 'EXTRAS' in self.sections and self.sections['EXTRAS'].debug:
                for element_basis_rank1_functions in b_basis.basis_rank1:
                    for basis_rank1_function in element_basis_rank1_functions:
                        basis_rank1_function.print()
            
                for element_basis_functions in b_basis.basis:
                    for basis_function in element_basis_functions:
                        basis_function.print()
                    
                for element_basis_rank1_functions in ctilde_basis.basis_rank1:
                    for basis_rank1_function in element_basis_rank1_functions:
                        basis_rank1_function.print()

                for element_basis_functions in ctilde_basis.basis:
                    for basis_function in element_basis_functions:
                        basis_function.print()

            
        except ImportError:
            raise RuntimeError("PyACE not available - cannot create basis")
        except Exception as e:
            self.pt.single_print(f"Error creating PyACE basis: {e}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create PyACE basis: {e}")
        
    # --------------------------------------------------------------------------------------------

    def _parse_embeddings(self, config):
        """Parse embedding configuration"""
        # JSON format
        embeddings = self.get_value("PYACE", "embeddings", "")
        
        # Check if it's already a dictionary (from API mode)
        if isinstance(embeddings, dict):
            self.embeddings = embeddings
        else:
            # It's a string, try to parse as JSON
            try:
                self.embeddings = json.loads(embeddings)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues (single quotes -> double quotes)
                try:
                    import ast
                    # Use ast.literal_eval for Python dict strings
                    self.embeddings = ast.literal_eval(embeddings)
                except (ValueError, SyntaxError):
                    raise RuntimeError(f"Error parsing embeddings: {e}")
        
    # --------------------------------------------------------------------------------------------
            
    def _parse_bonds(self, config):
        """Parse bond configuration"""
        # JSON format
        bonds = self.get_value("PYACE", "bonds", "")
        
        # Check if it's already a dictionary (from API mode)
        if isinstance(bonds, dict):
            self.bonds = bonds
        else:
            # It's a string, try to parse as JSON
            try:
                self.bonds = json.loads(bonds)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues (single quotes -> double quotes)
                try:
                    import ast
                    # Use ast.literal_eval for Python dict strings
                    self.bonds = ast.literal_eval(bonds)
                except (ValueError, SyntaxError):
                    raise RuntimeError(f"Error parsing bonds: {e}")
                    
        self.rcutfac = [b["rcut"] for b in self.bonds.values()]
        # print(f"*** self.nradmax {self.nradmax} self.lmax {self.lmax}")
     
        for b in self.bonds.values():
          b.update({
              'nradmax': self.nradmax,
              'nradbasemax': self.nradmax,
              'lmax': self.lmax,
          })
     
    # --------------------------------------------------------------------------------------------

    def _parse_functions(self, config):
        """Parse function configuration"""
        # JSON format
        functions = self.get_value("PYACE", "functions", "")
        
        # Check if it's already a dictionary (from API mode)
        if isinstance(functions, dict):
            self.functions = functions
        else:
            # It's a string, try to parse as JSON
            try:
                self.functions = json.loads(functions)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues (single quotes -> double quotes)
                try:
                    import ast
                    # Use ast.literal_eval for Python dict strings
                    self.functions = ast.literal_eval(functions)
                except (ValueError, SyntaxError):
                    raise RuntimeError(f"Error parsing functions: {e}")
                    
        self.nradmax = max(max(f["nradmax_by_orders"]) for f in self.functions.values())
        self.lmax    = max(max(f["lmax_by_orders"]) for f in self.functions.values())
                         
    # --------------------------------------------------------------------------------------------
    
    def _setup_type_mapping(self):
        """Setup type mapping and bond pairs for compatibility"""
        self.type_mapping = {}
        for i, element in enumerate(self.elements):
            self.type_mapping[element] = i + 1
    
    # --------------------------------------------------------------------------------------------
    
    def _create_coupling_coefficients_yace(self, output_filename="coupling_coefficients.yace"):
        """Create a coupling_coefficients.yace file using proper PyACE workflow
        
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
 
        @self.pt.rank_zero
        def _write_yace():
            try:
                # Save as .yace file
                self.ctilde_basis.save_yaml(output_filename)
            except Exception as e:
                self.pt.single_print(f"Error creating .yace file: {e}")
                import traceback
                self.pt.single_print(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to create .yace file: {e}")
                  
        _write_yace()
        self.pt.all_barrier()

    # --------------------------------------------------------------------------------------------

