
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
        
        # Get the configuration and ncoeff from PyAce section
        pyace_config = self.config.sections["PYACE"]
        
        if (self.config.sections["CALCULATOR"].nonlinear):
            a_width = pyace_config.ncoeff
        else:
            a_width = pyace_config.ncoeff * pyace_config.numtypes
            if not pyace_config.bzeroflag:
                a_width += pyace_config.numtypes
        
        return a_width
        
    def _set_computes(self):
        """Override parent method to create coupling coefficient file first"""
        
        # Create the coupling coefficient file if it doesn't exist
        coupling_filename = "coupling_coefficient.yace"
        
        try:
            # Check if file exists
            import os
            if not os.path.exists(coupling_filename):
                self.pt.single_print(f"Creating {coupling_filename} for LAMMPS compute pace...")
                
                # Get the PyAce configuration section
                pyace_config = self.config.sections["PYACE"]
                
                # Create the coupling coefficient file
                pyace_config.create_coupling_coefficient_yace(coupling_filename)
                self.pt.single_print(f"Successfully created {coupling_filename}")
            else:
                self.pt.single_print(f"Using existing {coupling_filename}")
        
        except Exception as e:
            self.pt.single_print(f"Error creating coupling coefficient file: {e}")
            raise RuntimeError(f"Failed to create {coupling_filename}: {e}")
        
        # Now call the parent method which uses the coupling file
        if not self._bikflag:
            base_pace = f"compute pace all pace {coupling_filename} 0 0"
        elif (self._bikflag and not self._dgradflag):
            base_pace = f"compute pace all pace {coupling_filename} 1 0"
        elif (self._bikflag and self._dgradflag):
            base_pace = f"compute pace all pace {coupling_filename} 1 1"
        
        self.pt.single_print(f"LAMMPS command: {base_pace}")
        self._lmp.command(base_pace)
        
 
    
    def create_coupling_coefficient_yace(self, output_filename="coupling_coefficient.yace"):
        """Create coupling_coefficient.yace file for LAMMPS compute pace
        
        This method uses the PyAce configuration to create a .yace file
        that can be used with LAMMPS compute pace command.
        
        Args:
            output_filename (str): Name of the output .yace file
            
        Returns:
            str: Path to the created .yace file
        """
        if not PYACE_AVAILABLE:
            raise RuntimeError("pyace not available")
            
        try:
            # Get the PyAce configuration section
            pyace_config = self.config.sections["PYACE"]
            
            # Use the PyAce section's method to create the coupling coefficient file
            coupling_file = pyace_config.create_coupling_coefficient_yace(output_filename)
            
            self.pt.single_print(f"Created coupling coefficient file: {coupling_file}")
            self.pt.single_print(f"This file can be used with LAMMPS compute pace")
            
            return coupling_file
            
        except Exception as e:
            self.pt.single_print(f"Error creating coupling_coefficient.yace: {e}")
            import traceback
            self.pt.single_print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create coupling_coefficient.yace: {e}")
    
    def get_lammps_usage_instructions(self, coupling_filename="coupling_coefficient.yace"):
        """Get instructions for using the coupling coefficient file with LAMMPS
        
        Args:
            coupling_filename (str): Name of the coupling coefficient file
            
        Returns:
            str: Instructions for LAMMPS integration
        """
        # Get element information from PyAce config
        pyace_config = self.config.sections["PYACE"]
        elements = pyace_config.elements
        
        instructions = f"""
LAMMPS Integration Instructions for PyACE:
==========================================

1. Use the coupling coefficient file: {coupling_filename}

2. In your LAMMPS input script, add:

   # Define atom types for elements: {' '.join(elements)}
   mass 1 <mass_of_{elements[0]}>"""
        
        if len(elements) > 1:
            for i, elem in enumerate(elements[1:], 2):
                instructions += f"\n   mass {i} <mass_of_{elem}>"
        
        instructions += f"""

   # Set up the ACE potential
   pair_style pace
   pair_coeff * * {coupling_filename} {' '.join(elements)}

   # Compute ACE descriptors
   compute pace_desc all pace {coupling_filename}
   
   # Output descriptors (optional)
   dump 1 all custom 1000 descriptors.dump id type c_pace_desc[*]

3. Element mapping:"""
        
        for i, elem in enumerate(elements, 1):
            instructions += f"\n   Type {i} = {elem}"
            
        instructions += f"""

Note: Make sure the atom types in your data file match this mapping.
"""
        
        return instructions
    
    def print_lammps_usage_instructions(self, coupling_filename="coupling_coefficient.yace"):
        """Print instructions for using the coupling coefficient file with LAMMPS"""
        self.pt.single_print(self.get_lammps_usage_instructions(coupling_filename))
    
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
    
 
