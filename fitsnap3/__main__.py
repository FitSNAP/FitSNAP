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

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    # fix stubs logic if mpi4py is installed in venv but running in serial
    stubs = 0 if comm.Get_size() > 1 else 1
except ModuleNotFoundError:
    stubs = 1
    comm = None


def main():
    # Instantiate single fitsnap instance for traditional flow of control.
    # This will create an internal parallel tools instance which will detect
    # availability of MPI for parallelization.
    try:
        fs = FitSnap(comm=comm)
        fs.scrape_configs(delete_scraper=True)
        if not "REAXFF" in fs.config.sections: fs.process_configs(delete_data=True) 
        # Good practice after a large parallel operation is to impose a barrier.
        fs.pt.all_barrier()
        fs.perform_fit()
        fs.write_output()
    except Exception as e:
        fs.pt.exception(e)


if __name__ == "__main__":
    main()
