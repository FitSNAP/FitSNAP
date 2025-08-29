#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import subprocess, os, warnings

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Command failed: {}\n{}".format(cmd, e.output))

def try_cmd(cmd):
    try:
        return run_cmd(cmd)
    except Exception:
        return ""

def mpi_flavor():
    s = try_cmd("mpirun -V").lower()
    if "open mpi" in s or "open-mpi" in s:
        return "openmpi"
    if "mpich" in s or "cray mpich" in s:
        return "mpich"
    if "intel" in s or "impi" in s or "intel(r) mpi" in s:
        return "intelmpi"
    warnings.warn("Unknown MPI; defaulting to 'openmpi' for BLACS inference.")
    return "openmpi"

def mpi_compile_link_flags():
    c = try_cmd("mpicc -showme:compile").split()
    l = try_cmd("mpicc -showme:link").split()
    if c or l:
        return c, l
    allf = try_cmd("mpicc -show").split()
    if not allf:
        return [], []
    comp, link = [], []
    for t in allf:
        if t.startswith("-I") or t.startswith("-D") or t.startswith("-std="):
            comp.append(t)
        else:
            link.append(t)
    return comp, link

def pkg_config_flags(*pc_names):
    for name in pc_names:
        cflags = try_cmd(f"pkg-config --cflags {name}").split()
        libs   = try_cmd(f"pkg-config --libs {name}").split()
        if not libs and not cflags:
            continue
        incs, libdirs, libnames, others, rpaths, ldflags_raw = [], [], [], [], [], []
        for f in cflags:
            if f.startswith("-I"):
                incs.append(f[2:])
            else:
                others.append(f)
        for f in libs:
            ldflags_raw.append(f)
            if f.startswith("-L"):
                libdirs.append(f[2:])
            elif f.startswith("-l"):
                libnames.append(f[2:])
            elif f.startswith("-Wl,-rpath,"):
                rpaths.append(f[len("-Wl,-rpath,"):])
        return incs, libdirs, libnames, others, rpaths, ldflags_raw
    return [], [], [], [], [], []

def unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out

mpi_kind = mpi_flavor()
mpi_cflags, mpi_ldflags = mpi_compile_link_flags()

include_dirs = [get_include()]
library_dirs = []
libraries    = []
extra_compile_args = []
extra_link_args    = []

# User overrides
ENV_LIBS    = os.environ.get("SCALAPACK_EXTRA_LIBS", "")
ENV_LDFLAGS = os.environ.get("SCALAPACK_EXTRA_LDFLAGS", "")
ENV_CFLAGS  = os.environ.get("SCALAPACK_EXTRA_CFLAGS", "")
if ENV_CFLAGS:
    extra_compile_args += ENV_CFLAGS.split()
if ENV_LDFLAGS:
    extra_link_args    += ENV_LDFLAGS.split()

# Prefer pkg-config
SCALAPACK_PKG = os.environ.get("SCALAPACK_PKG", "scalapack")
pc_incs, pc_libdirs, pc_libs, pc_cflags_other, pc_rpaths, pc_ldraw = pkg_config_flags(SCALAPACK_PKG)

if pc_libs:
    include_dirs += pc_incs
    library_dirs += pc_libdirs
    libraries    += pc_libs
    extra_compile_args += pc_cflags_other
    extra_link_args += [f for f in pc_ldraw if f.startswith("-Wl,")]
    extra_link_args += [f"-Wl,-rpath,{d}" for d in pc_rpaths]
else:
    # MKL fallback if MKLROOT is set
    MKLROOT = os.environ.get("MKLROOT", "")
    if MKLROOT:
        ilp64 = os.environ.get("MKL_ILP64", "0") == "1"
        intabi = "ilp64" if ilp64 else "lp64"

        # Add requested defines/arch flags
        if ilp64:
            extra_compile_args += ["-DMKL_ILP64"]
        # Force 64-bit only if user wants it; default toolchains on x86_64 are already 64-bit.
        if os.environ.get("FORCE_M64", "0") == "1":
            extra_compile_args += ["-m64"]

        blacs_choice = os.environ.get("MKL_BLACS", "").lower()
        if not blacs_choice:
            blacs_choice = "intelmpi" if mpi_kind == "intelmpi" else "openmpi"
        if blacs_choice not in ("openmpi", "intelmpi"):
            raise RuntimeError("MKL_BLACS must be 'openmpi' or 'intelmpi'.")

        mkl_libdir = os.path.join(MKLROOT, "lib", "intel64")
        library_dirs.append(mkl_libdir)
        libraries += [
            f"mkl_scalapack_{intabi}",
            f"mkl_blacs_{blacs_choice}_{intabi}",
            f"mkl_intel_{intabi}",
            "mkl_core",
            "mkl_sequential",
        ]
        # Your requested math/dlopen libs + typical MKL trailer, with rpath
        extra_link_args += ["-lm", "-ldl", "-lpthread", f"-Wl,-rpath,{mkl_libdir}"]
    else:
        # Generic last-resort
        libraries += ["scalapack"]
        if ENV_LIBS:
            libraries += ENV_LIBS.split()
        # Ensure basic math/dlopen are present if needed
        extra_link_args += ["-lm", "-ldl"]

# Merge MPI flags (“mpilinkargs” equivalent)
extra_compile_args = unique(mpi_cflags + extra_compile_args)
extra_link_args    = unique(mpi_ldflags + extra_link_args)

ext_modules = [
    Extension(
        "scalapack",
        sources=["scalapack.pyx"],
        include_dirs=unique(include_dirs),
        library_dirs=unique(library_dirs),
        libraries=unique(libraries),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    )
]

setup(
    name="scalapack_ext",
    version="0.1.1",
    description="Cython ScaLAPACK extension (pkg-config first, MKL-compatible fallback)",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
