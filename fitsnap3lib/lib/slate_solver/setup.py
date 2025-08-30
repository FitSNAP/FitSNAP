from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

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

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx", "slate_ridge.cpp"],
    include_dirs=[
        np.get_include(),
        f"{slate_dir}/include",
        "/usr/include/eigen3",  # Often needed for SLATE
    ],
    library_dirs=[f"{slate_dir}/lib", f"{slate_dir}/lib64"],  # lib64 for some systems
    libraries=["slate", "blaspp", "lapackpp", "blas", "lapack", "mpi"],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
    zip_safe=False,
)
