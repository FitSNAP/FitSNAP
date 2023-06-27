# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
# This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Charles Sievers (UC Davis, Sandia National Labs)
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

def initialize_fitsnap_run():
    try:
        import mpi4py as mpi4py
        from fitsnap3lib.parallel_tools import ParallelTools
        pt = ParallelTools()

    except ModuleNotFoundError:
        from fitsnap3lib.parallel_tools import ParallelTools
        pt = ParallelTools()

    except Exception as e:
        print("Trouble importing mpi4py package, exiting...")
        raise e

    pt.single_print("")
    pt.single_print("    ______ _  __  _____  _   __ ___     ____  ")
    pt.single_print("   / ____/(_)/ /_/ ___/ / | / //   |   / __ \ ")
    pt.single_print("  / /_   / // __/\__ \ /  |/ // /| |  / /_/ /")
    pt.single_print(" / __/  / // /_ ___/ // /|  // ___ | / ____/ ")
    pt.single_print("/_/    /_/ \__//____//_/ |_//_/  |_|/_/      ")
    pt.single_print("")
    pt.single_print("-----23Sep22------")

    try:
        pt.single_print("Reading input...")
        pt.all_barrier()
        from fitsnap3lib.io.input import Config
        config = Config()
        if (pt._rank==0):
            print(f"Hash: {config.hash}")
        pt.single_print("Finished reading input")
        pt.single_print("------------------")

        #from fitsnap3lib.io.input import output
    except Exception as e:
        pt.single_print("Trouble reading input, exiting...")
        pt.exception(e)

    try:
        pt.single_print("mpi4py version: ", mpi4py.__version__)

    except NameError:
        print("No mpi4py detected, using fitsnap stubs...")

    try:
        import numpy as np
        #output.screen("numpy version: ", np.__version__)
        pt.single_print
    except Exception as e:
        #output.screen("Trouble importing numpy package, exiting...")
        #output.exception(e)
        pt.single_print("Trouble importing numpy package, exiting...")
        pt.single_print(f"{e}")

    try:
        import scipy as sp
        pt.single_print("scipy version: ", sp.__version__)
    except Exception as e:
        pt.single_print("Trouble importing scipy package, exiting...")
        pt.single_print(e)

    try:
        import pandas as pd
        pt.single_print("pandas version: ", pd.__version__)
    except Exception as e:
        pt.single_print("Trouble importing pandas package, exiting...")
        pt.single_print(e)

    try:
        import lammps
        lmp = lammps.lammps()
        #print("LAMMPS version: ",lammps.__version__)
        pt.lammps_version = lammps.__version__
    except Exception as e:
        print("Trouble importing LAMMPS library, exiting...")
        raise e
    pt.single_print("-----------")
