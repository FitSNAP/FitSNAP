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

    #---------------------------------------------------------------------------
    # Genetic algorithm parameters
    
    # basic parameters
    population_size = 100 # <-- minimum 4 for testing, default is 100
    ngenerations = 50 # <-- minimum 4 for testing, default is 50

    # Advanced parameters, see libmod_optimize.py genetic_algorithm arguments for defaults
    # set exploration ranges for energies and forces
    my_w_ranges = [1.e-4,1.e-3,1.e-2,1.e-1,1,1.e1,1.e2,1.e3,1.e4]
    my_ef_ratios = [0.001,0.01,0.1,1,10,100,1000]
    
    # scaling of weights
    etot_weight = 1.0
    ftot_weight = 1.0
    
    # set parameters
    r_cross = 0.9
    r_mut = 0.1

    # set a score threshold for convergence 
    # TODO: need better explanation of what the score is
    # NOTE: could also have this operate on MAE or RMSE instead of score?
    conv_thr = 1.E-10

    # set a minimum value of ngenerations to run before checking for convergence (int(ngenerations/conv_check)).
    # example: if ngenerations = 100 and conv_check = 2, then convergence will only start being checked after half of the generations have been calculated (50 = 100/2).
    # make this number smaller to check for convergence earlier, and larger for later
    # tested defaults are between 2 and 3.
    conv_check = 2.

    # set designated group's force weights to zero (e.g. volume transformations)
    force_delta_keywords = []
    
    # write final best generation to FitSnap-compatible JSON dictionary. default is False
    write_to_json = True 
    
    # End genetic algorithm parameters
    #---------------------------------------------------------------------------
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
       lm_opt.genetic_algorithm(snap, 
                                population_size=population_size, 
                                ngenerations=ngenerations, 
                                my_w_ranges=my_w_ranges, 
                                my_ef_ratios=my_ef_ratios, 
                                etot_weight=etot_weight,
                                ftot_weight=ftot_weight,
                                r_cross=r_cross, 
                                r_mut=r_mut,
                                conv_thr=conv_thr, 
                                conv_check=conv_check, 
                                force_delta_keywords=force_delta_keywords,
                                write_to_json=write_to_json)
    snap.pt.single_print("Script complete, exiting")

if __name__ == "__main__":
    main()
