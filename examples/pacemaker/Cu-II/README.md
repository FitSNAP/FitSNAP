# Cu-II Pacemaker Example

This example demonstrates fitting a copper potential using a larger dataset (Cu-II) with the pacemaker-compatible PYACE calculator in FitSNAP.

## Data

The training data (`Cu_df2_1k.pkl.gzip`) contains 1000 copper structures with more complex basis functions and higher body orders compared to Cu-I.

## Key Differences from Cu-I

- Uses SBessel radial basis instead of ChebExpCos
- Higher maximum body orders: [15, 6, 4, 3, 2, 1] 
- Higher angular momentum: [0, 3, 3, 2, 2, 1]
- Includes test set splitting (10% of data)
- Uses power_order ladder scheme instead of body_order
- Automatic kappa (force/energy balance) determination
- Automatic inner cutoff based on minimal distances

## Configuration

This example uses hierarchical fitting with power order progression and includes validation testing.

## Usage

```bash
mpirun -np 4 python -m fitsnap3 Cu.in
```

## Expected Output

- More accurate fitted ACE potential due to larger training set
- Test set validation metrics
- Enhanced basis function complexity

## Notes

- Demonstrates advanced pacemaker features in FitSNAP
- Shows automatic parameter determination
- Includes proper train/test splitting
