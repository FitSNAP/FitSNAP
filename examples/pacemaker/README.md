# FitSNAP Pacemaker Examples

This directory contains examples demonstrating the use of FitSNAP's new PYACE calculator and pacemaker compatibility features.

## Overview

These examples show how to use FitSNAP with:
- **PYACE section**: New pacemaker-compatible input format
- **Pacemaker scraper**: Direct reading of `.pckl.gzip` data files  
- **Advanced ACE features**: Multi-element systems, hierarchical fitting, body-order specific functions

## Examples

### [Cu-I](Cu-I/) - Basic Single Element
- **System**: Copper (105 structures)
- **Features**: Basic ACE fitting, hierarchical ladder scheme
- **Complexity**: Beginner
- **Demonstrates**: Simple PYACE usage, body order progression

### [Cu-II](Cu-II/) - Advanced Single Element  
- **System**: Copper (1000 structures)
- **Features**: SBessel basis, higher body orders, test set validation
- **Complexity**: Intermediate
- **Demonstrates**: Advanced basis functions, automatic parameters

### [Ethanol](Ethanol/) - Multi-Element Molecular
- **System**: C2H6O molecules (1000 MD snapshots)
- **Features**: 3-element system, force-dominated training, core repulsion
- **Complexity**: Intermediate
- **Demonstrates**: Molecular systems, multi-element interactions

### [HEA](HEA/) - Complex Multi-Element
- **System**: 5-element High Entropy Alloy (Cr-Fe-Ni-Mn-Co)
- **Features**: Body-order specific functions, energy-based weighting
- **Complexity**: Advanced
- **Demonstrates**: Complex alloy systems, intelligent data selection

## Key Features Demonstrated

### PYACE Section Format
```ini
[PYACE]
elements = Cu
cutoff = 7.0
embeddings = {"ALL": {"npot": "FinnisSinclairShiftedScaled", ...}}
bonds = {"ALL": {"radbase": "ChebExpCos", ...}}
functions = {"UNARY": {"nradmax_by_orders": [15, 3, 2, 2, 1], ...}}
```

### Pacemaker Data Format
- Uses `scraper = PACEMAKER` 
- Reads `.pckl.gzip` files directly
- Automatic conversion from ASE atoms format
- Compatible with pacemaker datasets

### Advanced ACE Features
- **Hierarchical fitting**: Ladder schemes for systematic basis building
- **Body-order specific**: Different functions for different interaction orders
- **Multi-element**: Complex chemical space coverage
- **Energy weighting**: Intelligent structure selection

## Running Examples

All examples use the same basic command structure:

```bash
cd examples/pacemaker/[example_name]
mpirun -np 4 python -m fitsnap3 [input_file].in
```

### Prerequisites
- FitSNAP with PYACE support
- `pyace` Python package installed
- `pandas` and `numpy` for data handling

### Expected Outputs
- LAMMPS-compatible ACE potential files
- Training metrics and validation results
- Descriptor and truth data dumps for analysis

## Notes

- These examples require the new pacemaker scraper and PYACE calculator
- Data files (`.pckl.gzip`) should be copied from the pacemaker examples
- FIT section comments show original pacemaker parameters for reference
- All examples use SVD solver - can be replaced with other FitSNAP solvers

## Conversion from Pacemaker

The examples demonstrate how pacemaker YAML inputs convert to FitSNAP format:

| Pacemaker YAML | FitSNAP PYACE |
|----------------|---------------|
| `elements: [Cu]` | `elements = Cu` |
| `cutoff: 7` | `cutoff = 7.0` |
| `embeddings: ALL: {...}` | `embeddings = {"ALL": {...}}` |
| `bonds: ALL: {...}` | `bonds = {"ALL": {...}}` |
| `functions: UNARY: {...}` | `functions = {"UNARY": {...}}` |

This provides a smooth migration path from pacemaker workflows to FitSNAP.
