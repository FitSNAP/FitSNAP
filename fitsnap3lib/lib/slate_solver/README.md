## SLATE-based Multi-node Ridge Solver Implementation

### Overview
Implemented a new distributed Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale) to enable multi-node scaling for FitSNAP.

### Architecture
The solver follows a minimal, clean architecture:
```
ridge_slate.py → slate_wrapper.pyx → slate_ridge.cpp → SLATE library
```

### Implementation Details

#### Core Components
- **`solvers/ridge_slate.py`**: Main solver class that inherits from FitSNAP's base Solver
  - Handles data distribution across nodes
  - Manages training/testing splits  
  - Applies weights and computes local portions of A^T A and A^T b
  - Calls SLATE wrapper for distributed solving

- **`lib/slate_solver/`**: Minimal SLATE wrapper (4 essential files)
  - `slate_wrapper.pyx`: Cython bridge between Python and C++
  - `slate_ridge.cpp`: C++ implementation using SLATE's distributed matrix operations
  - `setup.py`: Build configuration
  - `__init__.py`: Module initialization

#### Key Features
- Distributed ridge regression solving: `(A^T A + α*I)x = A^T b`
- MPI-based parallelization across multiple nodes
- Automatic 2D process grid optimization for SLATE
- Configurable tile size for SLATE's block-cyclic distribution
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
solver = RidgeSlate

[RIDGE]
alpha = 1e-6

[SLATE]  # optional
tile_size = 256
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
- Uses Cholesky factorization (posv) for positive definite systems
- Efficient MPI collective operations for matrix reductions
- Designed for scaling to large HPC clusters

### Technical Details
- Matrix reduction: Uses MPI_Allreduce to aggregate local A^T A and A^T b across all nodes
- Process grid: Automatically determines optimal 2D grid layout (as square as possible)
- Memory efficiency: Operates on local matrix portions to minimize memory footprint per node
- Ridge regularization: Adds α to diagonal after global reduction for numerical stability
