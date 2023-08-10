"""
Genetic algorithm optimization of SNAP potential

Serial usage:

    python libmod_optimize.py # --fitsnap_in Ta-example.in --optimization_style genetic_algorithm

Parallel usage:

    mpirun -n 2 python libmod_optimize.py # --fitsnap_in Ta-example.in --optimization_style genetic_algorithm
"""

import os, time, argparse, warnings
import numpy as np
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
import libmod_optimize as lm_opt
# NOTE warnings have been turned off for zero divide errors!
warnings.filterwarnings('ignore')

def main():
    # set up mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # parse command line args
    fs_input="SNAP_Ta.in"
    # fs_input="ACE_Ta.in"
    optimization_style = "genetic_algorithm"
    popsize, ngen = 4, 4 # default values: 50, 100; testing values:  4, 4
    smartweights_override = False

    parser = argparse.ArgumentParser(description='FitSNAP example.')
    parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=fs_input)
    parser.add_argument("--optimization_style", help="optimization algorithm: 'simulated_annealing' or 'genetic_algorithm' ", default=optimization_style)
    args = parser.parse_args()

    optimization_style = args.optimization_style

    settings = args.fitsnap_in

    # Create a FitSnap instance using the communicator and settings:
    snap = FitSnap(settings, comm=comm, arglist=["--overwrite"])
    snap.pt.single_print("FitSNAP input script:", args.fitsnap_in)

    # prepare snap config for genetic algorithm
    lm_opt.prep_fitsnap_input(snap)

    # run initial fit (calculate descriptors)
    snap.scrape_configs()
    snap.process_configs()
    snap.pt.all_barrier()
    snap.perform_fit()
    fit1 = snap.solver.fit
    errs1 = snap.solver.errors

    # NOTE: when using MPI, the following lines may throw a warning error but the program will still run
    rmse_e = errs1.iloc[:,2].to_numpy()
    rmse_counts = errs1.iloc[:,0].to_numpy()
    rmse_eat = rmse_e[0]
    rmse_fat = rmse_e[1]
    rmse_tot = rmse_eat + rmse_fat
    snap.pt.single_print(f'Initial fit:\n\trsme energies: {rmse_eat}\n\trsme forces: {rmse_fat}\n\t total:{rmse_tot}')

    snap.solver.fit = None

    snap.pt.single_print("FitSNAP optimization algorithm: ",args.optimization_style)

    if optimization_style == 'simulated_annealing':
        lm_opt.sim_anneal(snap)
    elif optimization_style == 'genetic_algorithm':
       lm_opt.genetic_algorithm(snap, population_size=popsize, ngenerations=ngen, write_to_json=True)
    snap.pt.single_print("Script complete, exiting")

if __name__ == "__main__":
    main()
