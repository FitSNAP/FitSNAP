from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
import shutil
import glob

# --- MPI detection ---
try:
    mpicc_path = shutil.which("mpicc")
    if mpicc_path is None:
        raise RuntimeError
    mpi_compile_flags = subprocess.check_output(
        ["mpicc", "--showme:compile"]
    ).decode().strip().split()
    mpi_link_flags = subprocess.check_output(
        ["mpicc", "--showme:link"]
    ).decode().strip().split()
    print(f"Found MPI compiler: {mpicc_path}")
except Exception:
    print("Warning: mpicc not found, using defaults")
    mpi_compile_flags = []
    mpi_link_flags = ["-lmpi"]

mpi_include_dirs = [f[2:] for f in mpi_compile_flags if f.startswith("-I")]

# --- SLATE detection (add default locations) ---
def find_slate_dir():
    # Candidate prefixes to probe
    candidates = [
        os.environ.get("SLATE_DIR", ""),
        os.path.expanduser("~/.local"),
        "/usr/local",
        "/usr",
        "/opt/slate",
        "/opt/local",                 # MacPorts
        "/usr/local/opt/slate",       # Homebrew (Intel macOS)
        "/opt/homebrew/opt/slate",    # Homebrew (Apple Silicon)
    ]

    # Best-effort Spack search if SPACK_ROOT is set
    spack_root = os.environ.get("SPACK_ROOT")
    if spack_root:
        pattern = os.path.join(spack_root, "opt", "spack", "**", "include", "slate")
        for d in glob.glob(pattern, recursive=True):
            prefix = d[:-len("/include/slate")]
            candidates.insert(1, prefix)  # put near front

    for path in candidates:
        if not path:
            continue
        if os.path.exists(os.path.join(path, "include", "slate")):
            print(f"Found SLATE in: {path}")
            return path

    print("Warning: SLATE not found in standard locations. Set SLATE_DIR if needed.")
    return os.path.expanduser("~/.local")

slate_dir = find_slate_dir()

# --- Include and library dirs ---
include_dirs = [
    np.get_include(),
    os.path.join(slate_dir, "include"),
    "/usr/include/eigen3",
] + mpi_include_dirs

library_dirs = [
    os.path.join(slate_dir, "lib"),
    os.path.join(slate_dir, "lib64"),
]

libraries = ["slate", "mpi"]

# --- OpenMP (conditional) ---
extra_compile_args = ["-std=c++17", "-O3"] + [
    f for f in mpi_compile_flags if not f.startswith("-I")
]
extra_link_args = [f for f in mpi_link_flags if not f.startswith("-l")]

is_macos = (sys.platform == "darwin")
libomp_prefix = None
if is_macos:
    try:
        libomp_prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
    except Exception:
        libomp_prefix = None

if is_macos and libomp_prefix:
    include_dirs.append(os.path.join(libomp_prefix, "include"))
    library_dirs.append(os.path.join(libomp_prefix, "lib"))
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-L" + os.path.join(libomp_prefix, "lib"), "-lomp"]
elif is_macos and not libomp_prefix:
    print("Note: libomp not found via Homebrew; building without OpenMP on macOS.")
else:
    # Non-macOS: GCC/Clang with libgomp/llvm-omp
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

print(f"Libraries to link: {libraries}")
print(f"Library directories: {library_dirs}")
print(f"Extra compile args: {extra_compile_args}")
print(f"Extra link args: {extra_link_args}")

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx", "slate_ridge.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=cythonize([ext], language_level="3"),
    zip_safe=False,
)
