# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3 
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
# 
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
# 
# #### Original author: 
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps 
# 
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
# 
# #### Additional authors (alphabetical): 
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
#     
# <!-----------------END-HEADER------------------------------------->


import itertools
import ctypes
import copy

import numpy as np

from . import geometry


def extract_compute_np(lmp,name,compute_style,result_type,array_shape):
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute stores floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is 
    skipped and a python float is returned.

    From LAMMPS/src/library.cpp:
    style = 0 for global data, 1 for per-atom data, 2 for local data
    type = 0 for scalar, 1 for vector, 2 for array

    """
    ptr = lmp.extract_compute(name, compute_style, result_type) 
    if result_type == 0: return ptr # No casting needed, lammps.py already works
    if result_type == 2: ptr = ptr.contents
    total_size = np.prod(array_shape)
    buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * total_size))
    array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
    array_np.shape = array_shape
    return array_np

def extract_commands(string):
    return [x for x in string.splitlines() if x.strip() != '']

def set_style(lmp, numtypes):
    style_commands = extract_commands("""
    pair_style hybrid/overlay lj/cut ${rcutfac} zbl ${zblcutinner} ${zblcutouter}
    pair_coeff * * lj/cut 0.0 1.0
    """)
    for i, j in itertools.combinations_with_replacement(range(1, numtypes + 1), 2):
        print("Setting types",i,j)
        # Note: Doubled {{ }} syntax escapes python string formating
        this_str = f"pair_coeff {i} {j} zbl ${{zblz{i}}} ${{zblz{j}}}"
        # print("This string:",this_str)
    #    lmp.command(this_str)


def lammps_variables(bispec_options):
    d = {k: bispec_options[k] for k in
         ["rcutfac",
          "rfac0",
          "rmin0",
          "twojmax"]}
    d.update(
        {
            (k + str(i + 1)): bispec_options[k][i]
            for k in ["wj", "radelem"] #"zblz", "wj", "radelem"
            for i, v in enumerate(bispec_options[k])
        })
    return d


def set_box(lmp, lattice_array, numtypes):
    lmp.command("boundary p p p")

    ((ax, bx, cx),
     (ay, by, cy),
     (az, bz, cz)) = lattice_array

    assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
        "Cell not normalized for lammps!"
    region_command = \
        f"region pybox prism 0 {ax:20.20g} 0 {by:20.20g} 0 {cz:20.20g} {bx:20.20g} {cx:20.20g} {cy:20.20g}"
    lmp.command(region_command)
    lmp.command(f"create_box {numtypes} pybox")


def set_variables(lmp, **lmp_variable_args):
    for k, v in lmp_variable_args.items():
        lmp.command(f"variable {k} equal {v}")


def set_computes(lmp, bispec_options):
    numtypes = bispec_options["numtypes"]
    radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
    wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

    kw_options = {
        k: bispec_options[v]
        for k, v in
        {
            "rmin0": "rmin0",
            "bzeroflag": "bzeroflag",
            "quadraticflag": "quadraticflag",
            "switchflag": "switchflag",
        }.items()
        if v in bispec_options
    }
    kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
    kwargs = " ".join(kw_substrings)

    # everything is handled by LAMMPS compute snap

    base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
    command = f"{base_snap} {radelem} {wj} {kwargs}"
    lmp.command(command)

def extract_computes(lmp, num_atoms, n_coeff,num_types, compute_dbvb, TrainFile):

    lmp_atom_ids = lmp.numpy.extract_atom_iarray("id", num_atoms).flatten()
    assert np.all(lmp_atom_ids==1+np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

    # Extract positions
    lmp_pos = lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
    # Extract types
    lmp_types = lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).flatten()
    lmp_volume = lmp.get_thermo("vol")

    res = {
        "AtomTypes": lmp_types,
        "Positions": lmp_pos,
        "Volume": lmp_volume,
        }

    # Extract SNAP data, including reference potential data

    nrows_energy = 1
    ndim_force = 3
    nrows_force = ndim_force*num_atoms
    ndim_virial = 6
    nrows_virial = ndim_virial
    nrows_snap = nrows_energy+nrows_force+nrows_virial
    ncols_bispectrum = num_types*n_coeff
    ncols_reference = 1
    ncols_snap = ncols_bispectrum + ncols_reference

    lmp_snap = extract_compute_np(lmp,"snap",0,2,(nrows_snap,ncols_snap))

    irow = 0
    res["b_sum"] = lmp_snap[irow,:ncols_bispectrum]
    res["b_sum"].shape = (num_types,n_coeff)
    icolref = ncols_bispectrum
    lmp_sume = lmp_snap[irow,icolref]
    if (np.isinf(lmp_sume)).any() or (np.isnan(lmp_sume)).any():
        print("Inf or NaN Energy returned from LAMMPS :",TrainFile)
    res["ref_Energy"] = lmp_sume
    irow += nrows_energy

    if compute_dbvb:
        res["db_atom"] = lmp_snap[irow:irow+nrows_force,:ncols_bispectrum]
        res["db_atom"].shape = (num_atoms,ndim_force,num_types,n_coeff)
        res["ref_Forces"] = lmp_snap[irow:irow+nrows_force,icolref]
        res["ref_Forces"].shape = (num_atoms,ndim_force)
        irow += nrows_force

        res["vb_sum"] = lmp_snap[irow:irow+nrows_virial,:ncols_bispectrum]
        res["vb_sum"].shape = (ndim_virial,num_types,n_coeff)
        res["ref_Stress"] = lmp_snap[irow:irow+nrows_virial,icolref]

    # Copy arrays because lammps will reuse memory.
    return {k: copy.copy(v) for k, v in res.items()}

def create_atoms(lmp, numtypes, type_dict, types, positions):
    for i, (a_t, (a_x, a_y, a_z)) in enumerate(zip(types, positions)):
        a_t = type_dict[a_t]
        lmp.command(f"create_atoms {a_t} single {a_x:20.20g} {a_y:20.20g} {a_z:20.20g} remap yes")
    n_atoms = int(lmp.get_natoms())
    assert i + 1 == n_atoms, f"Atom counts don't match when creating atoms: {i+1}, {n_atoms}"

def create_spins(lmp, bispec_options, spins):
    for i, (s_mag, s_x, s_y, s_z) in enumerate(spins):
        lmp.command(f"set atom {i+1} spin {s_mag:20.20g} {s_x:20.20g} {s_y:20.20g} {s_z:20.20g} ")
    n_atoms = int(lmp.get_natoms())
    assert i + 1 == n_atoms, f"Atom counts don't match when assigning spins: {i+1}, {n_atoms}"

def create_charge(lmp, bispec_options, charges):
    for i, q in enumerate(charges):
        lmp.command(f"set atom {i+1} charge {q:20.20g} ")
    n_atoms = int(lmp.get_natoms())
    assert i + 1 == n_atoms, f"Atom counts don't match when assigning charge: {i+1}, {n_atoms}"

def compute_lammps(lmp, data, bispec_options):
    lmp.command("clear")
    lmp.command("units " + bispec_options["units"])
    lmp.command("atom_style " + bispec_options["atom_style"])

    lmp_setup = extract_commands("""
        atom_modify map array sort 0 2.0
        box tilt large""")
    for line in lmp_setup:
        lmp.command(line)

    set_box(lmp, data["Lattice"], numtypes=bispec_options["numtypes"])

    create_atoms(lmp, bispec_options["numtypes"], bispec_options["type_mapping"], data["AtomTypes"], data["Positions"])

    if (bispec_options["atom_style"]=="spin"):
        create_spins(lmp, bispec_options, data["Spins"])
    if (bispec_options["atom_style"]=="charge"):
        create_charge(lmp, bispec_options, data["Charges"])

    set_variables(lmp, **lammps_variables(bispec_options))

    for line in bispec_options["pair_func"]:
        lmp.command(line.lower())
    set_computes(lmp, bispec_options)

    lmp.command("mass * 1.0e-20")
    lmp.command("neighbor 1.0e-20 nsq")
    lmp.command("neigh_modify one 10000")
    lmp.command("run 0")

    computed_data = extract_computes(lmp,
                                     num_atoms=data["NumAtoms"],
                                     n_coeff=bispec_options["n_coeff"],
                                     num_types=bispec_options["numtypes"],
                                     compute_dbvb=bispec_options["compute_dbvb"],
                                     TrainFile=data["File"]
                                     )
    # raises AssertionError if lammps moved the atoms by anything other than a lattice vector
    geometry.check_coords(data["Lattice"], computed_data["Positions"], data["Positions"])
    # raise Assertion Error if the cell size changed from original QM to computed in LAMMPS
    geometry.check_volume(data["QMLattice"],computed_data["Volume"])

    return computed_data
