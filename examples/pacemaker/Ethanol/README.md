# Ethanol Pacemaker Example

This example demonstrates fitting a multi-element molecular potential for ethanol (C2H6O) using the pacemaker-compatible PYACE calculator in FitSNAP.

## Data

The training data (`ethanol.pckl.gzip`) contains 1000 ethanol molecule MD snapshots from the revMD17 dataset.

## Multi-Element Features

- **Elements**: Carbon (C), Oxygen (O), Hydrogen (H)
- **Molecular interactions**: C-C, C-O, C-H, O-O, O-H, H-H bonds
- **Short cutoff**: 4.0 Å suitable for molecular systems
- **Force-dominated fitting**: kappa=0.99 (99% force weight)

## Key Features

- Demonstrates multi-element ACE potential for organic molecules
- Uses core-repulsion for short-range interactions
- High force weighting appropriate for molecular dynamics
- Multiple fitting cycles for convergence
- Body order ladder scheme progression

## Configuration

- **Cutoff**: 4.0 Å (appropriate for molecular systems)
- **Basis**: ChebExpCos radial functions
- **Body orders**: Up to rank 5 (1-5 body interactions) 
- **Core repulsion**: Included for all bond types

## Usage

```bash
mpirun -np 4 python -m fitsnap3 Ethanol.in
```

## Expected Output

- Multi-element ACE potential suitable for ethanol MD simulations
- High-accuracy force predictions for molecular motions
- Potential transferable to similar organic molecules

## Notes

- Demonstrates FitSNAP's multi-element capabilities via PYACE
- Shows molecular system parameterization
- Includes advanced pacemaker features like core repulsion
- Force-centric training for accurate dynamics
