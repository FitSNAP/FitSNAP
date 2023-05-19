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
#     Drew Rohskopf (Sandia National Labs)
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
# <!-----------------END-HEADER------------------------------------->

from fitsnap3lib.fitsnap import FitSnap


def main():
    # Instantiate single fitsnap instance for traditional flow of control.
    # This will create an internal parallel tools instance which will detect
    # availability of MPI for parallelization.
    snap = FitSnap()
    snap.scrape_configs(delete_scraper=True)
    snap.process_configs(delete_data=True)
    # Good practice after a large parallel operation is to impose a barrier.
    snap.pt.all_barrier()
    snap.perform_fit()
    snap.write_output()
    """
    # TODO: Might be cleaner ways to output errors when doing massively parallel runs.
    except Exception as e:
        #output.exception(e)
        print(str(e))
        snap.pt.single_print(f"{e}")
    """


if __name__ == "__main__":
    main()
