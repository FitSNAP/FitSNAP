from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

# Try to get MPI compiler and flags
try:
    mpicc = subprocess.check_output(['which', 'mpicc']).decode().strip()
    mpi_compile_flags = subprocess.check_output(['mpicc', '--showme:compile']).decode().strip().split()
    mpi_link_flags = subprocess.check_output(['mpicc', '--showme:link']).decode().strip().split()
    print(f"Found MPI compiler: {mpicc}")
except:
    print("Warning: mpicc not found, using defaults")
    mpi_compile_flags = []
    mpi_link_flags = ['-lmpi']

# Extract include dirs from MPI flags
mpi_include_dirs = [flag[2:] for flag in mpi_compile_flags if flag.startswith('-I')]

# Try different possible SLATE installation locations
slate_dir = None
for path in [os.environ.get('SLATE_DIR', ''), 
             os.path.expanduser("~/.local"),
             "/usr/local",
             "/opt/slate"]:
    if path and os.path.exists(os.path.join(path, "include", "slate")):
        slate_dir = path
        print(f"Found SLATE in: {slate_dir}")
        break

if slate_dir is None:
    print("Warning: SLATE not found. Set SLATE_DIR environment variable.")
    slate_dir = os.path.expanduser("~/.local")  # fallback

# Build include directories list
include_dirs = [
    np.get_include(),
    f"{slate_dir}/include",
    "/usr/include/eigen3",  # Often needed for SLATE
    "/opt/homebrew/Cellar/libomp/20.1.8/include",  # OpenMP headers
] + mpi_include_dirs

# Library directories - just SLATE's location
library_dirs = [f"{slate_dir}/lib", f"{slate_dir}/lib64", "/opt/homebrew/Cellar/libomp/20.1.8/lib"]

# Libraries - only SLATE and MPI
libraries = ["slate", "mpi", "omp"]

print(f"Libraries to link: {libraries}")
print(f"Library directories: {library_dirs}")

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx", "slate_ridge.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"] + [f for f in mpi_compile_flags if not f.startswith('-I')],
    extra_link_args=[f for f in mpi_link_flags if not f.startswith('-l')],
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
    zip_safe=False,
)
