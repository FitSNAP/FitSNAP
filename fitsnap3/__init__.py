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
#     Charles Sievers (Sandia National Labs)
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

try:
    import mpi4py as mpi4py
    from fitsnap3.parallel_tools import pt

except ModuleNotFoundError:
    from fitsnap3.parallel_tools import pt

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
pt.single_print("-----------")

try:
    pt.single_print("Reading input...")
    pt.all_barrier()
    from fitsnap3.io.input import config
    pt.single_print("Finished reading input")
    from fitsnap3.io.output import output
except Exception as e:
    pt.single_print("Trouble reading input, exiting...")
    pt.exception(e)

try:
    output.screen("mpi4py version: ", mpi4py.__version__)

except NameError:
    print("No mpi4py detected, using fitsnap stubs...")

try:
    import numpy as np
    output.screen("numpy version: ", np.__version__)
except Exception as e:
    output.screen("Trouble importing numpy package, exiting...")
    output.exception(e)

try:
    import scipy as sp
    output.screen("scipy version: ", sp.__version__)
except Exception as e:
    output.screen("Trouble importing scipy package, exiting...")
    output.exception(e)

try:
    import pandas as pd
    output.screen("pandas version: ", pd.__version__)
except Exception as e:
    output.screen("Trouble importing pandas package, exiting...")
    output.exception(e)

# try:
#     import lammps
#     print("LAMMPS version: ",lammps.version())
# except Exception as e:
#     print("Trouble importing LAMMPS library, exiting...")
#     raise e
output.screen("-----------")
