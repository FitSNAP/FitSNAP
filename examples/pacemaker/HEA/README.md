# HEA (High Entropy Alloy) Pacemaker Example

This example demonstrates fitting a complex multi-element High Entropy Alloy potential using the pacemaker-compatible PYACE calculator in FitSNAP.

## Data

The training data (`HEA_randII_example.pckl.gzip`) contains structures from a 5-element High Entropy Alloy system.

## Multi-Element System

- **Elements**: Chromium (Cr), Iron (Fe), Nickel (Ni), Manganese (Mn), Cobalt (Co)
- **Complex interactions**: All possible pair, triple, quadruple, and quintuple interactions
- **Hierarchical body orders**: Different complexity for different body orders
- **Advanced weighting**: Energy-based weighting policy for structure selection

## Key Features

- **5-element system**: Demonstrates FitSNAP's multi-element capabilities
- **Body-order specific functions**: Different basis sets for different body orders
- **Energy-based weighting**: Intelligent structure selection based on energy criteria
- **Convex hull weighting**: Uses convex hull for reference energy determination
- **Limited dataset**: Only 200 structures selected for fitting

## Configuration Highlights

- **QUATERNARY functions**: Specialized 4-body interaction terms
- **QUINARY functions**: Full 5-body interaction terms  
- **Energy weighting policy**: DElow=1.0 eV, DEup=10.0 eV
- **Force-dominated**: kappa=0.95 (95% force weight)
- **Small batch size**: batch_size=20 for memory management

## Usage

```bash
mpirun -np 4 python -m fitsnap3 HEA.in
```

## Expected Output

- Multi-element ACE potential for HEA system
- Advanced body-order interactions
- Optimized structure selection metrics

## Notes

- Most complex pacemaker example demonstrating advanced features
- Shows body-order specific function definitions
- Demonstrates intelligent data selection strategies
- Suitable for complex metallurgical applications
