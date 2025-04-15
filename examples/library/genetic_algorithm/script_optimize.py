"""
Genetic algorithm to optimize group weights in FitSNAP fits

Serial usage:

    python script_optimize.py 

Parallel usage:

    mpirun -n 2 python script_optimize.py 

When using MPI, a quick note on number of cores P: 
- For now, it's best to use an even number of cores P.
- If your population_size setting is smaller than P, it will be increased to P (to avoid running empty cores per generation).
- If you're using MPI but don't want to run the GA in parallel, set the optional genetic_algorithm argument `parallel_population = False.`
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
    parser = argparse.ArgumentParser(description='FitSNAP example.')
    parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default="SNAP_Ta.in")
    parser.add_argument("--optimization_style", help="optimization algorithm: 'simulated_annealing' or 'genetic_algorithm' ", default="genetic_algorithm")
    parser.add_argument("--perform_initial_fit", action=argparse.BooleanOptionalAction) ##default is false
    args = parser.parse_args()

    fs_input=args.fitsnap_in
    # fs_input="SNAP_Ta.in"
    # fs_input="ACE_Ta.in"
    optimization_style = args.optimization_style
    perform_initial_fit = args.perform_initial_fit

    # verbose flag prints out much more info per tested fit
    verbose = False
    #---------------------------------------------------------------------------
    # Genetic algorithm parameters
    
    # basic parameters
    population_size = 30 # <-- minimum 4 for testing, default is 100
    ngenerations = 20 # <-- minimum 4 for testing, default is 50

    # Advanced parameters, see libmod_optimize.py genetic_algorithm arguments for defaults
    # set exploration ranges for energies and forces
    my_w_ranges = [1.e-4,1.e-3,1.e-2,1.e-1,1,1.e1,1.e2,1.e3,1.e4]
    my_ef_ratios = [0.001,0.01,0.1,1,10,100,1000]
    my_es_ratios = [0.001,0.01,0.1,1,10,100,1000]
    
    # scaling of weights relative to each other
    etot_weight = 1.0
    ftot_weight = 1.0
    stot_weight = 1.0
    
    # set parameters
    r_cross = 0.9
    r_mut = 0.1

    # set a score threshold for convergence 
    # TODO: need an explanation of what the score is
    conv_thr = 1.E-10

    # set a fraction of ngenerations to check for score convergence
    # example: if ngenerations = 100 and conv_check = 0.5, then convergence will only start being checked at generation 51, after half of the generations have been calculated. 
    # make this number smaller to check for convergence earlier, and larger for later.
    # tested defaults are between 0.33 to 0.5.
    conv_check = 0.5

    # set designated group's force weights to zero (e.g. volume transformations)
    force_delta_keywords = []
    stress_delta_keywords = []
    
    # write final best generation to FitSnap-compatible JSON dictionary. default is False
    write_to_json = False 
    
    # use_initial_weights reads from current file, default is False
    # If True, these are multiplied by the exploration ranges
    use_initial_weights = False 
    
    # End genetic algorithm parameters
    #---------------------------------------------------------------------------

    # Create a FitSnap instance using the communicator and settings:
    snap = FitSnap(fs_input, comm=comm, arglist=["--overwrite"])
    snap.pt.single_print("FitSNAP input script:", fs_input)

    # prepare snap config for genetic algorithm
    lm_opt.prep_fitsnap_input(snap)

    # run initial fit (calculate descriptors)
    snap.scrape_configs()
    snap.process_configs()
    snap.pt.all_barrier()

    # use can decide whether or not to perform fit with initial fs_input settings
    # this is useful if one would like to compare the original model with a resulting model
    if perform_initial_fit:
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

    additional_cost_functions = []
    additional_cost_weights = []
    
    ##### optional section for extra cost functions
    if False: #Switch this to true to add lattice constant and elastic constants (BCC crystal structure) to your objective function
        ## WARNING! This feature will not work properly in parallel runs yet!
        ## This will need to be set to wherever your lammps executable is to run lammps calculations
        lmp = "/usr/workspace/wsb/logwill/code/james_lammps_branch/lammps_compute_PACE/build_v6_see_list/lmp"
        lammps_elastic_input_script = "./in.elastic"
        ## this bit reads in an output of a lammps calculation with a different model form as the truth. normally you would just set these values to exp or DFT data.
        elastic_truth_output_filepath = "./truth_output"
        #TODO: update to match new class format
        reader = lm_opt.ElasticPropertiesFromLAMMPS(lmp, lammps_elastic_input_script, truth_values=False, existing_output_path=elastic_truth_output_filepath)
        elastic_truth_vals = reader.output_vals()  ##this is where you would normally just set the vals. format: [lattice constant, C11, C12, C44]
        del reader
        #TODO: update to match new class format
        additional_cost_functions.append(lm_opt.ElasticPropertiesFromLAMMPS(lmp, lammps_elastic_input_script, truth_values=elastic_truth_vals, existing_output_path=False))
        lattice_weight = 1.0
        C11_weight = 1.0
        C12_weight = 100.0
        C44_weight = 100.0
        additional_cost_weights.append([lattice_weight, C11_weight, C12_weight, C44_weight])

    # perform optimization algorithms 
    snap.pt.single_print("FitSNAP optimization algorithm: ",optimization_style)
    if optimization_style == 'genetic_algorithm':
       lm_opt.genetic_algorithm(snap, 
                                population_size=population_size, 
                                ngenerations=ngenerations, 
                                my_w_ranges=my_w_ranges, 
                                my_ef_ratios=my_ef_ratios, 
                                my_es_ratios=my_es_ratios, 
                                etot_weight=etot_weight,
                                ftot_weight=ftot_weight,
                                stot_weight=stot_weight,
                                r_cross=r_cross, 
                                r_mut=r_mut,
                                conv_thr=conv_thr, 
                                conv_check=conv_check, 
                                force_delta_keywords=force_delta_keywords,
                                stress_delta_keywords=force_delta_keywords,
                                write_to_json=write_to_json,
                                use_initial_weights_flag=use_initial_weights,
                                additional_cost_functions = additional_cost_functions,
                                additional_cost_weights = additional_cost_weights,
                                verbose=verbose)
   
    # TODO implement other SNAP optimizations
    # elif optimization_style == 'simulated_annealing':
    #     # tODO integrate MPI and other new fixes into simulated annealing
    #     lm_opt.sim_anneal(snap)
    # elif optimization_style == 'latin_hypercube':
    #     lm_opt.latin_hypercube_sample(snap)
    snap.pt.single_print("Script complete, exiting")

if __name__ == "__main__":
    main()
