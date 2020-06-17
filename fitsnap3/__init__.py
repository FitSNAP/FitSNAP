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
print("")
print("    ______ _  __  _____  _   __ ___     ____  ")
print("   / ____/(_)/ /_/ ___/ / | / //   |   / __ \ ")
print("  / /_   / // __/\__ \ /  |/ // /| |  / /_/ /")
print(" / __/  / // /_ ___/ // /|  // ___ | / ____/ ")
print("/_/    /_/ \__//____//_/ |_//_/  |_|/_/      ")
print("")
print("-----------")
try:
    import numpy as np
    print("numpy version: ",np.__version__)
except Exception as e:
    print("Trouble importing numpy package, exiting...")
    raise e

try:
    import pandas as pd
    print("pandas version: ",pd.__version__)
except Exception as e:
    print("Trouble importing pandas package, exiting...")
    raise e

try:
    import sklearn as skl
    print("scikit-learn version: ",skl.__version__)
except Exception as e:
    print("Trouble importing scikit-learn package, exiting...")
    raise e

try:
    import scipy as sp
    print("scipy version: ",sp.__version__)
except Exception as e:
    print("Trouble importing scipy package, exiting...")
    raise e

try:
    import tqdm
    print("tqdm version: ",tqdm.__version__)
except Exception as e:
    print("Trouble importing tqdm package, exiting...")
    raise e

try:
    import natsort
    print("natsort version: ",natsort.__version__)
except Exception as e:
    print("Trouble importing natsort package, exiting...")
    raise e
#try:
#    import lammps
#    print("LAMMPS version: ",lammps.version())
#except Exception as e:
#    print("Trouble importing LAMMPS library, exiting...")
#    raise e
print("-----------")

from . import bispecopt, deploy, geometry, linearfit, runlammps, scrape, serialize
