# Cu-I Pacemaker Example

This example demonstrates fitting a simple copper potential using the pacemaker-compatible PYACE calculator in FitSNAP.

## Data

The training data (`Cu_df1_A1_A2_A3_EV_elast_phon.pckl.gzip`) contains 105 copper structures from various configurations including:
- A15, BCC, FCC displaced structures
- Elastic deformations
- Phonon calculations

## Configuration

This example uses hierarchical fitting (ladder scheme) with body order progression to systematically build up the ACE basis complexity.

## Usage

```bash
mpirun -np 4 python -m fitsnap3 Cu.in
```

The training will proceed through ladder steps, adding new basis functions according to body order progression.

## Expected Output

- Fitted ACE potential in LAMMPS format
- Training metrics and error analysis
- Potential files ready for LAMMPS simulations

## Notes

- Uses the new PYACE section for pacemaker compatibility
- Employs pacemaker scraper for .pckl.gzip data format
- Demonstrates conversion from pacemaker YAML to FitSNAP input format
