## SLATE-based Multi-node Ridge Solver Implementation

### Overview
Implemented a new distributed Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale) to enable multi-node scaling for FitSNAP.

### Architecture
The solver follows a minimal, clean architecture:
```
slate.py → slate_wrapper.pyx → slate.cpp → SLATE library
```

### Implementation Details

#### Core Components
- **`solvers/slate.py`**: Main solver class that inherits from FitSNAP's base Solver
  - Handles data distribution across nodes
  - Manages training/testing splits  
  - Applies weights
  - Calls SLATE wrapper for distributed solving

- **`lib/slate_solver/`**: Minimal SLATE wrapper (4 essential files)
  - `slate_wrapper.pyx`: Cython bridge between Python and C++
  - `slate.cpp`: C++ implementation using SLATE's distributed matrix operations
  - `setup.py`: Build configuration
  - `__init__.py`: Module initialization

#### Key Features
- Distributed ridge regression using augmented QR
- MPI-based parallelization across multiple nodes
- 1D process grid for SLATE
- Supports weighted least squares and training/testing splits
- Compatible with FitSNAP's existing parallel infrastructure

### Building
```bash
cd fitsnap3lib/lib/slate_solver
pip install -e .
# or alternatively: python setup.py build_ext --inplace
```

### Usage
In FitSNAP configuration:
```ini
[SOLVER]
solver = SLATE

[SLATE] 
alpha = 1e-6
```

### Requirements
- SLATE library installed (typically in `~/.local`)
- BLASPP and LAPACKPP (SLATE dependencies)
- BLAS/LAPACK libraries
- MPI implementation (OpenMPI/MPICH)
- C++17 compiler
- Cython

### Performance
- Leverages SLATE's optimized distributed linear algebra routines
- Uses Augmented QR for numerical stability
- Efficient MPI collective operations for matrix reductions
- Designed for scaling to large HPC clusters

### Technical Details
- Process grid: 1D grid layout
- Memory efficiency: Operates on local matrix portions to minimize memory footprint per node
- Ridge regularization: Adds α to diagonal after global reduction for numerical stability
- Tile size: currently not optimized, one tile per mpi rank 
