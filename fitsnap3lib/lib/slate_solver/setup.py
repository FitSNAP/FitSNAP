from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

slate_dir = os.path.expanduser("~/.local")

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx", "slate_ridge.cpp"],
    include_dirs=[
        np.get_include(),
        f"{slate_dir}/include",
        "/usr/include/eigen3",  # Often needed for SLATE
    ],
    library_dirs=[f"{slate_dir}/lib"],
    libraries=["slate", "blaspp", "lapackpp", "blas", "lapack", "mpi"],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
    zip_safe=False,
)
