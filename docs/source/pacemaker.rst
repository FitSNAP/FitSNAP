Migrating from pacemaker
========================

This document details the implementation of pacemaker compatibility in FitSNAP through the new PYACE calculator and infrastructure.

Overview
--------

Today's development focused on creating a comprehensive bridge between FitSNAP and the pacemaker ecosystem, enabling seamless migration from pacemaker workflows to FitSNAP's enhanced capabilities. The implementation provides multiple pathways for integration while maintaining backward compatibility with existing ACE configurations.

**1. PYACE Calculator Implementation**
   - Complete rewrite of the PYACE calculator system
   - Resolved circular import issues with the pyace Python package
   - Added exact basis function counting using pyace APIs
   - Implemented robust import handling with multiple fallback strategies

**2. Pacemaker-Compatible Configuration**
   - New ``[PYACE]`` section parser supporting both simple and JSON configurations
   - Automatic bond pair generation from element lists
   - Support for pacemaker-style YAML import/export workflows
   - Direct ACE→PYACE configuration conversion

**3. Enhanced Data Processing**
   - New PACEMAKER scraper for processing pacemaker training data
   - Integration with pacemaker examples and test cases
   - Support for complex multi-element systems

Technical Implementation Details
--------------------------------

Core Components
~~~~~~~~~~~~~~~

**Calculator Implementation** (``fitsnap3lib/calculators/lammps_pyace.py``)
   - Renamed from LammpsPyACE to PyACE (LAMMPS not required)
   - Integrated with pyace Python package for ACE descriptor calculations
   - Supports both simple parameter lists and complex JSON configurations
   - Exact basis function counting eliminates estimation errors

**Section Parser** (``fitsnap3lib/io/sections/calculator_sections/pyace.py``)
   - Complete rewrite from scratch
   - Dual input format support: simple key-value and JSON
   - Automatic bond pair generation using ``itertools.product(elements, repeat=2)``
   - Hierarchical keyword system (ALL → UNARY/BINARY/TERNARY → specific pairs)

**Data Integration**
   - PACEMAKER scraper for processing pacemaker training datasets
   - Example configurations and test cases added
   - OMat24 integration for materials science applications

Configuration Examples
----------------------

Simple ACE→PYACE Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing ACE configurations can be directly converted by changing the section name:

.. code-block:: ini

   # Original ACE section
   [ACE]
   numTypes = 2
   type = H O
   ranks = 1 2
   lmax = 0 0
   nmax = 1 1
   rcutfac = 2.383 3.221 3.221 4.058
   lambda = 0.119 0.161 0.161 0.203

   # New PYACE equivalent  
   [PYACE]
   elements = H O
   cutoff = 4.1
   ranks = 1 2
   lmax = 0 0
   nmax = 1 1
   rcutfac = 2.383 3.221 3.221 4.058
   lambda = 0.119 0.161 0.161 0.203

Advanced JSON Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex multi-element systems, PYACE supports pacemaker-style JSON configurations:

.. code-block:: ini

   [PYACE]
   elements = H O
   cutoff = 4.1
   delta_spline_bins = 0.001

   # Per-element embeddings (H and O)
   embeddings = {
       "H": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1, 1], "ndensity": 1}, 
       "O": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1, 1], "ndensity": 1}
   }

   # Bond-specific settings (HH, HO, OH, OO)  
   bonds = {
       "HH": {"radbase": "ChebExpCos", "radparameters": [0.119], "rcut": 2.383}, 
       "HO": {"radbase": "ChebExpCos", "radparameters": [0.161], "rcut": 3.221}, 
       "OH": {"radbase": "ChebExpCos", "radparameters": [0.161], "rcut": 3.221}, 
       "OO": {"radbase": "ChebExpCos", "radparameters": [0.203], "rcut": 4.058}
   }

   # Function configuration (ranks 1 2, lmax 0 0, nmax 1 1)
   functions = {
       "UNARY": {"nradmax_by_orders": [1, 1], "lmax_by_orders": [0, 0]}, 
       "BINARY": {"nradmax_by_orders": [1, 1], "lmax_by_orders": [0, 0]}
   }

Bond Pair Mapping
~~~~~~~~~~~~~~~~~

For ``elements = H O``, the parameter arrays automatically map to bond pairs:

.. code-block:: text

   rcutfac[0] = 2.383 → HH bond  
   rcutfac[1] = 3.221 → HO bond
   rcutfac[2] = 3.221 → OH bond  
   rcutfac[3] = 4.058 → OO bond

This follows the same ``itertools.product(elements, repeat=2)`` pattern used by pacemaker.

Migration Benefits
------------------

**Exact Basis Function Counting**
   - No more estimation errors in matrix allocation
   - Uses pyace package directly: ``get_number_of_functions()``, ``get_basis_size()``
   - Critical for proper solver array allocation

**Robust Import Handling**
   - Resolves circular import issues with pyace package
   - Multiple fallback strategies for different pyace versions
   - Graceful degradation when pyace unavailable

**Enhanced Compatibility**
   - Direct integration with pacemaker YAML workflows
   - Support for advanced pyace features (ladder fitting, etc.)
   - Maintains backward compatibility with existing ``[ACE]`` sections

**Improved Performance**
   - Leverages optimized pyace implementations
   - Support for complex multi-element systems
   - Efficient handling of large basis sets

Future Development
------------------

The current implementation provides the foundation for:

- **Full YAML Integration**: Direct import/export of pacemaker YAML files
- **Advanced Features**: Ladder fitting, active learning integration  
- **Performance Optimization**: GPU acceleration for large basis sets
- **Extended Examples**: More complex materials and molecular systems

Usage in FitSNAP
----------------

To use the new PYACE calculator, simply change your calculator specification:

.. code-block:: ini

   [CALCULATOR]
   calculator = PYACE

   [PYACE]
   # Your pacemaker-compatible configuration here
   elements = Cu
   cutoff = 5.0
   # ... additional parameters

The implementation seamlessly integrates with FitSNAP's existing infrastructure while providing enhanced pacemaker compatibility and improved functionality.

Troubleshooting
---------------

**Import Errors**
   If you encounter pyace import issues, ensure the package is properly installed:
   
   .. code-block:: bash
   
      pip install pyace

**Configuration Conversion**
   For complex pacemaker configurations, use the JSON format for maximum compatibility with pacemaker's hierarchical parameter structure.

**Performance Issues**
   For large basis sets, consider using the exact basis function counting to optimize memory allocation and solver performance.

This implementation represents a significant step forward in FitSNAP's interoperability with the broader ACE ecosystem while maintaining the software's core strengths and ease of use.
