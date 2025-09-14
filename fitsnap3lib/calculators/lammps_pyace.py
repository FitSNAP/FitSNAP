
import numpy as np
#from fitsnap3lib.calculators.calculator import Calculator
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


class LammpsPyace(LammpsPace):
    """
    Calculator using pyace basis in [PYACE] with LAMMPS compute pace
    """
    
    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, calculator_section="PYACE")
        
        self._data = {}
        self._i = 0
        self._row_index = 0
        
        # Initialize pyace specific parameters
        self.ace_basis = None
        
        
    def get_width(self):
        """Get width of descriptor vector for PYACE calculator"""
        
        if not PYACE_AVAILABLE:
            raise RuntimeError("pyace not available")
        
 
    
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
    
 
