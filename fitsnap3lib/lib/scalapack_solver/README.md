## FitSNAP ScaLAPACK library

This library will generate a shared object library for multinode solving capabilities.
Files in this directory are written by Charlie A. Sievers and updated by @alphataubio (2025/08):


## ✅ **Fixed NumPy Deprecation Issues**

**In `scalapack.pyx`:**
- ✅ Added proper NumPy C API initialization: `np.import_array()`  
- ✅ Replaced ALL deprecated `.data` attribute access with modern `np.PyArray_DATA()`
- ✅ Added proper Cython memoryview declarations for better performance
- ✅ Used explicit Cython NumPy interface (`cimport numpy as np`)

## ✅ **Modernized Build System**  

**Created `pyproject.toml`:**
- ✅ PEP 518 compliant modern Python packaging
- ✅ Proper build system requirements and dependencies
- ✅ Modern project metadata

**Updated `setup.py`:**
- ✅ Replaced deprecated `distutils` with modern `setuptools`  
- ✅ Used `cythonize()` with modern compiler directives
- ✅ Added performance optimizations (boundscheck=False, etc.)
- ✅ Proper dependency management

## 🚀 **How to Build Now**

**Modern way (recommended):**
```bash
# Clean previous builds first
rm -rf build/ dist/ *.egg-info/ scalapack.c *.so

# Modern installation
pip install -e .
```

**Legacy way (still works):**
```bash
# Clean and rebuild  
rm -rf build/ scalapack.c *.so
python setup.py build_ext --inplace
```

## 🎯 **Results**

- ❌ **No more NumPy deprecation warnings!**
- ❌ **No more setup.py deprecation warnings** (when using modern methods)
- ✅ **Better performance** due to optimized Cython compilation
- ✅ **Future-proof** with modern Python packaging standards
- ✅ **Fully compatible** with Python 3.8+ and modern NumPy


# ScaLAPACK Cython Extension - Modern Build

This ScaLAPACK Cython extension has been modernized to use:
- Modern setuptools instead of deprecated distutils
- Current NumPy C API (no deprecated API warnings)
- Modern Cython compilation with proper memoryviews and type annotations
- PEP 518 compliant build system with pyproject.toml

## Modern Build Methods (Recommended)

### Method 1: Using pip (recommended)
```bash
# Install in development mode
pip install -e .

# Or build wheel and install
pip install build
python -m build
pip install dist/*.whl
```

### Method 2: Using build tool
```bash
pip install build
python -m build
```

## Legacy Build Methods (still supported)

### Clean build
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/ scalapack.c *.so

# Build extension in place
python setup.py build_ext --inplace

# Install (discouraged, use pip instead)
python setup.py install
```

## Key Modernizations Made

### NumPy API
- ✅ Uses `np.PyArray_DATA()` instead of deprecated `.data` attribute
- ✅ Proper NumPy C API initialization with `np.import_array()`
- ✅ Modern Cython memoryview syntax for better performance
- ✅ Explicit type declarations for all NumPy arrays

### Build System
- ✅ Uses `setuptools` instead of deprecated `distutils`
- ✅ Uses `cythonize()` with modern compiler directives
- ✅ PEP 518 compliant with `pyproject.toml`
- ✅ Proper dependency management
- ✅ Modern Python packaging standards

### Environment Variables (unchanged)
- `SCALAPACK_PKG`: pkg-config package name (default: "scalapack")
- `MKLROOT`: Intel MKL root directory for MKL-based builds
- `MKL_ILP64`: Set to "1" for ILP64 interface
- `MKL_BLACS`: "openmpi" or "intelmpi" for BLACS implementation
- `SCALAPACK_EXTRA_LIBS`: Additional libraries to link
- `SCALAPACK_EXTRA_LDFLAGS`: Additional linker flags
- `SCALAPACK_EXTRA_CFLAGS`: Additional compiler flags

## Notes
- No more NumPy deprecation warnings!
- No more setup.py deprecation warnings when using modern build methods
- Better performance due to optimized Cython compilation
- Fully compatible with Python 3.8+ and modern NumPy versions


(older readme follows)


### Building this library:

This sublibrary of FitSNAP is compiled and therefore depends on specific machine and module
settings. ScaLAPACK libraries require Intel MKL and specific library include settings for 
different machines are found online at the 
[Intel MKL link advisor tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html).

Once you have the linking and include options for your specific setup, include these options 
in the `scl_lib` list located in `setup.py`. Then we are ready to build.  

To build this library, use the following commands in this directory:

    python setup.py build_ext --inplace
    python setup.py install

These commands will compile the shared object library and then move the build to your 
environment site packages, respectively.

### Files in this Directory

#### setup.py 

Python file containing the setup configuration. (Cython setuptools)

#### scalapack.h

C header file containing ScaLAPACK function declarations.

#### scalapack.pxy

ScaLAPACK cython file.

#### Scalapack.pxd

Cython header file containing ScaLAPACK function declarations.

#### scalapack.py

Python file containing ScaLAPACK lstsq solver.

### Important installation information

Requires mpi4py python package, OpenMP, and ScaLAPACK libraries (often by loading a MKL module).

This package has been tested using GNU compilers, OpenMPI, and Intel MKL modules. Please 
find specific library linking flags for your specific module versions using the Intel 
MKL link advisor tool.
