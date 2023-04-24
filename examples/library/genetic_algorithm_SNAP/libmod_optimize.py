"""
Genetic algorithm optimization of SNAP potential

Usage:

    #parallel
    mpirun -n 2 python3 libmod_optimize.py --fitsnap_in Ta-example.in --optimization_style genetic_algorithm
    #serial
    python3 libmod_optimize.py --fitsnap_in Ta-example.in --optimization_style genetic_algorithm
"""

import numpy as np
from mpi4py import MPI
import argparse
import gc
import pandas as pd
import warnings
import time
# NOTE warnings have been turned off for zero divide errors!
warnings.filterwarnings('ignore')

# parse command line args

parser = argparse.ArgumentParser(description='FitSNAP example.')
parser.add_argument("--fitsnap_in", help="FitSNAP input script.", default=None)
parser.add_argument("--optimization_style", help="optimization algorithm: 'simulated_annealing' or 'genetic_algorithm' ", default=None)
args = parser.parse_args()

optimization_style = args.optimization_style

comm = MPI.COMM_WORLD

# import parallel tools and create pt object
# this is the backbone of FitSNAP
from fitsnap3lib.parallel_tools import ParallelTools
pt = ParallelTools(comm=comm)
# Config class reads the input
from fitsnap3lib.io.input import Config
config = Config(arguments_lst = [args.fitsnap_in, "--overwrite"])
# create a fitsnap object
from fitsnap3lib.fitsnap import FitSnap
snap = FitSnap()
# import other necessaries to run basic example
from fitsnap3lib.io.output import output
from fitsnap3lib.initialize import initialize_fitsnap_run

pt.single_print("FitSNAP input script:",args.fitsnap_in)

# run initial fit (calculate descriptors)
snap.scrape_configs()
snap.process_configs()
pt.all_barrier()
snap.perform_fit()
fit1 = snap.solver.fit
errs1 = snap.solver.errors


# get groups and weights 
gtks = config.sections["GROUPS"].group_table.keys()
gtks = list(gtks)
pt.single_print('groups',gtks)

size_b = np.shape(pt.fitsnap_dict['Row_Type'])[0]
grouptype = pt.fitsnap_dict['Groups'].copy()
rowtype = pt.fitsnap_dict['Row_Type'].copy()


rmse_e = errs1.iloc[:,2].to_numpy()
rmse_counts = errs1.iloc[:,0].to_numpy()
rmse_eat = rmse_e[0]
rmse_fat = rmse_e[1]
rmse_tot = rmse_eat + rmse_fat
pt.single_print('Initial fit',rmse_eat,rmse_fat,rmse_tot)

snap.solver.fit = None

pt.single_print("FitSNAP optimization algorithm:",args.optimization_style)
class CostObject:
    @pt.rank_zero
    def __init__(self):
        self.conts = None
        self.cost = 999999999.9999
        self.unweighted = np.array([])
        self.weights = np.array([])

    @pt.rank_zero
    def cost_contributions(self,cost_conts,weights):
        cost_conts = np.array(cost_conts) 
        weights = np.array(weights)
        wc_conts = weights*cost_conts
        self.unweighted = cost_conts
        self.weights = weights
        self.conts = wc_conts

    @pt.rank_zero
    def add_contribution(self,costi,weighti=1.0):
        try:
            if self.conts == None:
                cond = True
            else:
                cond = False
        except ValueError:
            cond = False
        if cond:
            cost_conts = np.array([costi])
            weights = np.array([weighti])
            wc_conts = weights*cost_conts
            self.conts = wc_conts
            self.weights = np.append(self.weights, weighti)
            self.unweighted = np.append(self.unweighted, costi)
        else:
            self.conts = np.append(self.conts,costi*weighti)
            self.weights = np.append(self.weights,weighti)
            self.unweighted = np.append(self.unweighted,costi)

    @pt.rank_zero
    def evaluate_cost(self):
        costi = np.sum(self.conts)
        self.cost = costi
        return costi

class HyperparameterStruct:
    @pt.rank_zero
    def __init__(self,
            ne,
            nf,
            eranges,
            ffactors,):
        self.ne = ne
        self.nf = nf
        self.nh = ne+nf
        self.erangesin = eranges
        self.ffactorsin = ffactors
        self.set_eranges()
        self.set_ffactors()

    @pt.rank_zero
    def set_eranges(self):
        if len(self.erangesin) != self.ne and len(self.erangesin) == 1:
            self.eranges = self.erangesin * self.ne
        elif len(self.erangesin) == self.ne:
            self.eranges = self.erangesin
        else:
            raise ValueError('incorrect number of values for energy group weight ranges, ether specify range for each group or sepecify one range to be applied to all groups')

    @pt.rank_zero
    def set_ffactors(self):
        if len(self.ffactorsin) != self.nf and len(self.ffactorsin) == 1:
            self.ffactors = self.ffactorsin * self.nf
        elif len(self.ffactorsin) == self.nf:
            self.ffactors = self.ffactorsin
        else:
            raise ValueError('incorrect number of values for force group weight ratios, ether specify range for each group or sepecify one range to be applied to all groups')
    
    @pt.rank_zero
    def random_params(self,inputseed=None):
        if inputseed != None:
            np.random.seed(inputseed)
        self.eweights = np.random.rand(self.ne) * np.array([np.random.choice(self.eranges[ihe]) for ihe in range(self.ne)])
        f_factors =  np.random.rand(self.nf) * np.array([np.random.choice(self.ffactors[ihf]) for ihf in range(self.nf)])
        self.fweights = self.eweights * f_factors
        return np.append(self.eweights,self.fweights)

# tournament selection
@pt.rank_zero
def tournament_selection(population, scores, k=3, inputseed=None):
    if inputseed != None:
        np.random.seed(inputseed)
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

class Selector:
    @pt.rank_zero
    def __init__(self,selection_style = 'tournament'):
        self.selection_style = selection_style
        self.set_selector()

    @pt.rank_zero
    def set_selector(self):
        if self.selection_style == 'tournament':
            self.selection = tournament_selection

# when 2 parent creatures love eachother very much, they make/adopt 2 children creatures
# this does not include cross-over for stresses yet.
@pt.rank_zero
def crossover(p1, p2, ne, w_combo_delta=np.array([]), ef_rat_delta=np.array([]), inputseed=None):
    if inputseed != None:
        np.random.seed(inputseed)
    c1, c2 = p1.copy(), p2.copy()
    c1e,c1f = tuple(c1.reshape(2,ne))
    c2e,c2f = tuple(c2.reshape(2,ne))
    p1e,p1f = tuple(p1.reshape(2,ne))
    p2e,p2f = tuple(p2.reshape(2,ne))
    # select crossover point that corresponds to a certain group
    pt = np.random.randint(1, ne-2)
    # only perform crossover between like hyperparameters (energy and energy then force and force, etc.)
    if np.shape(w_combo_delta)[0] != 0:
        c1e = np.append(p1e[:pt], p2e[pt:])*w_combo_delta
        c1f = np.append(p1f[:pt], p2f[pt:])*ef_rat_delta
        c2e = np.append(p2e[:pt], p1e[pt:])*w_combo_delta
        c2f = np.append(p2f[:pt], p1f[pt:])*ef_rat_delta
    else:
        c1e = np.append(p1e[:pt] , p2e[pt:])
        c1f = np.append(p1f[:pt] , p2f[pt:])
        c2e = np.append(p2e[:pt] , p1e[pt:])
        c2f = np.append(p2f[:pt] , p1f[pt:])
    c1 = np.append(c1e,c1f)
    c2 = np.append(c2e,c2f)
    return [c1, c2]


@pt.rank_zero
def update_weights(test_w_combo, test_ef_rat, test_virial_w=(1.e-8,), gtks = gtks, size_b = size_b, grouptype = grouptype):
    if len(test_virial_w) == 1:
        tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_virial_w[0]} for ig,gkey in enumerate(gtks)  }
    elif len(test_virial_w) == len(test_ef_rat):
        tstwdct = {gkey:{'eweight':test_w_combo[ig], 'fweight':test_w_combo[ig]*test_ef_rat[ig], 'vweight':test_virial_w[ig]} for ig,gkey in enumerate(gtks)  }
    else:
        raise IndexError("not enough virial indices per energy and force indices")

    #loop through data and update pt shared array based on group type
    for index_b in range(size_b):
        gkey = grouptype[index_b]
        if pt.fitsnap_dict['Row_Type'][index_b] == 'Energy':
            pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['eweight']
        elif pt.fitsnap_dict['Row_Type'][index_b] == 'Force':
            pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['fweight']
        elif pt.fitsnap_dict['Row_Type'][index_b] == 'Stress':
            pt.shared_arrays['w'].array[index_b] = tstwdct[gkey]['vweight']

@pt.rank_zero
def ediff_cost(fit,g1,g2,target,grouptype=grouptype,rowtype=rowtype):
    # provides 'cost' for energy differences for amatrix entries:
    #  (\beta * a1 - \beta * a2) - target
    # where target is some target energy difference between structures from group 1 and group 2
    # NOTE that a1 and a2 Amatrix entries are identified by indexing. You 
    # MUST put only one structure per group to use in this objective function
    indexg1 = grouptype.index(g1)
    indexg2 = grouptype.index(g2)
    assert rowtype[indexg1] == 'Energy',"not accessing energy row for group %s" % g1
    assert rowtype[indexg2] == 'Energy',"not accessing energy row for group %s" % g2
    a1 = pt.shared_arrays['a'].array[indexg1].copy()
    a2 = pt.shared_arrays['a'].array[indexg2].copy()

    e1 = np.sum(a1*fit)
    e2 = np.sum(a2*fit)
    diff = e1-e2

    #pt.single_print(e1,e2,diff,target)

    return np.abs(diff - target)

#NOTE fit_and_cost will likely need to be modified to print current fit if
# other objective functions are to be added. 
@pt.rank_zero
def fit_and_cost(snap,costweights):
    etot_weight, ftot_weight = tuple(costweights)
    #clear old fit and solve test fit
    snap.solver.fit = None
    snap.perform_fit()
    fittst = snap.solver.fit
    errstst = snap.solver.errors
    rmse_tst = errstst.iloc[:,2].to_numpy()
    rmse_countstst = errstst.iloc[:,0].to_numpy()

    #pt.single_print(errstst)
    rmse_eattst = rmse_tst[0]
    rmse_fattst = rmse_tst[1]
    CO = CostObject()
    CO.add_contribution(rmse_eattst,etot_weight)
    CO.add_contribution(rmse_fattst,ftot_weight)
    # commented examples on how to use energy differences in the objective function
    # a SINGLE structure is added to two new fitsnap groups, the corresponding energy
    # difference between group 1 and group 2 is given as the target (in eV) (NOT eV/atom)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_1',g2='H_above_W_1',target=-777.91314380000-(-775.10721929000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)
    #obj_ads = ediff_cost(fittst,g1='H_ads_W_2',g2='H_above_W_2',target=-781.72430764000-(-781.66087408000),grouptype=grouptype)
    #CO.add_contribution(obj_ads,etot_weight * 1)
    cost = CO.evaluate_cost()
    del CO
    return cost

@pt.rank_zero
def seed_maker(mc,mmax = 1000000000,use_saved_seeds=True):
    if use_saved_seeds:
        try:
            seeds = np.load('seeds.npy')
            if np.shape(seeds)[0] < mc:
                pt.single_print('potentially not enough seeds for this run, appending more')
                seeds = np.append(seeds , np.random.randint(0,mmax, mc- (np.shape(seeds)[0])  ))
                np.save('seeds.npy',seeds)
            else:
                seeds = seeds
        except FileNotFoundError:
            seeds = np.random.randint(0,mmax,mc)
            np.save('seeds.npy',seeds)
    else:
        seeds = np.random.randint(0,mmax,mc)
        np.save('seeds.npy',seeds)
    return seeds

@pt.rank_zero
def mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng,w_combo_delta=np.array([]),ef_rat_delta=np.array([]),apply_random=True,full_mutation=False):
    if type(current_w_combo) == tuple:
        current_w_combo = np.array(current_w_combo)
    if type(current_ef_rat) == tuple:
        current_ef_rat = np.array(current_ef_rat)
    if full_mutation:
        if apply_random:
            test_w_combo = np.random.rand()*np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.rand()*np.random.choice(my_ef_ratios,ng)
        else:
            test_w_combo = np.random.choice(my_w_ranges,ng)
            test_ef_rat = np.random.choice(my_ef_ratios,ng)
    else:
        test_w_combo = current_w_combo.copy()
        test_ef_rat = current_ef_rat.copy()
        test_w_ind = np.random.choice(range(ng))
        if apply_random:
            plusvsprd =  1 #TODO implement addition/product steps after constraining min/max weights
            if plusvsprd:
                test_w_combo[test_w_ind] = np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] = np.random.rand() * np.random.choice(my_ef_ratios)
            else:
                test_w_combo[test_w_ind] *= np.random.rand() * np.random.choice(my_w_ranges)
                test_ef_rat[test_w_ind] *= np.random.rand() * np.random.choice(my_ef_ratios)

        else:
            test_w_combo[test_w_ind] = np.random.choice(my_w_ranges)
            test_ef_rat[test_w_ind] = np.random.choice(my_ef_ratios)
    if np.shape(w_combo_delta)[0] != 0:
        return test_w_combo * w_combo_delta, test_ef_rat*ef_rat_delta
    else:
        return test_w_combo,test_ef_rat


@pt.rank_zero
def print_final(ew_frcrat_final,train_sz=1.0,test_sz=0.0):
    ew_final,frcrat_final = ew_frcrat_final
    pt.single_print('Best group weights\n')
    for idi,dat in enumerate(gtks):
        en_weight = ew_final[idi]
        frc_weight = ew_final[idi]*frcrat_final[idi]
        pt.single_print('%s       =  %1.2f      %1.2f      %.16E      %.16E      1.E-12' % (dat, train_sz,test_sz,en_weight,frc_weight))

#-----------------------------------------------------------------------
# begin the primary optimzation functions
#-----------------------------------------------------------------------
time1 = time.time()
@pt.rank_zero
def sim_anneal():
    #---------------------------------------------------------------------------
    #Begin optimization hyperparameters
    etot_weight = 1.0
    ftot_weight = 1.5
    rmse_tot = 500
    # sampling magnitudes per hyperparameter
    my_w_ranges = [1.e-3,1.e-2,1.e-1,1.e0,1.e1,1.e2,1.e3]
    my_ef_ratios = [0.1,1,10]

    # Artificial temperatures
    betas = [1.e0,1.e1,1.e2,1.e3,1.e4]
    # Max number of steps per artificial temperature
    count_per_beta = [400,400,600,1000,1000]
    # threshhold for convergence of cost function
    thresh = 0.005

    seedpad = 50
    #build seeds (uses saved seeds by default)
    countmaxtot = int(np.sum(count_per_beta))
    seedsi = seed_maker(countmaxtot + seedpad)


    #End optimization hyperparameters
    #---------------------------------------------------------------------------
    
    tot_count = 0
    #threshold for cost function before accepting model
    current_w_combo = [1.e0]*len(gtks)
    current_ef_rat = [10]*len(gtks)
    tot_count = 0
    apply_random = True # flag to select a single random hyperparam to step rather than stepping all hyperparams
    naccept = 0
    np.random.seed(seedsi[tot_count])
    # loop over fictitious temperatures
    for ibeta,beta in enumerate(betas):
        count = 0
        naccepti = 0
        maxcount = count_per_beta[ibeta]
        # propose trial weights while counts are below maximums and 
        # objective function is above threshhold
        while count <= maxcount and rmse_tot >= thresh:

            if tot_count <= 5: # allow for large steps early in simulation
                test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=True)
            else:
                test_w_combo, test_ef_rat = mutation(current_w_combo,current_ef_rat,my_w_ranges,my_ef_ratios,ng=len(gtks),apply_random=True,full_mutation=False)

            update_weights(test_w_combo, test_ef_rat, gtks = gtks, size_b = size_b, grouptype = grouptype)

            rmse_tottst = fit_and_cost(snap,[etot_weight,ftot_weight])

            delta_Q = rmse_tottst - rmse_tot
            boltz = np.exp(-beta*delta_Q)
            rndi = np.random.rand()
            logical = rndi <= boltz
            if logical:
                naccept += 1
                naccepti +=1
                rmse_tot = rmse_tottst
                current_w_combo = test_w_combo
                current_ef_rat = test_ef_rat

            meta = (tuple(list(current_w_combo)),) + (tuple(list(current_ef_rat)),)
            count += 1
            pt.single_print('beta',beta,'count',count,' accept ratio for current beta %f' % (naccepti/count) ,meta,boltz,rmse_tottst,rmse_tot)
            tot_count += 1
            np.random.seed(seedsi[tot_count])
    # write output for optimized potential
    print_final(meta)
    time2 = time.time()
    pt.single_print('Total optimization time,', time2 - time1, 'total number of fits', tot_count)
    snap.write_output()


@pt.rank_zero
def genetic_algorithm():
    #---------------------------------------------------------------------------
    #Begin optimization hyperparameters

    # population of generations
    population_size = 26
    # number of generations
    ngenerations = 300

    countmaxtot = int(population_size*(ngenerations+2))
    seedsi = seed_maker(countmaxtot)

    #number of hyperparameters:
    # num of energy group weights
    ne = len(gtks)
    # num of force group weights
    nf = ne
    # total
    nh = ne + nf

    # weights for energy and force rmse in the optimizer cost function
    etot_weight = 1.0
    ftot_weight = 5.0
    # allowed scaling factors for energy weights
    my_w_ranges = [1.e-3,1.e-2,1.e-1,1,1.e1,1.e2,1.e3]
    eranges = [my_w_ranges]
    # allowed scaling factors for force weights
    my_ef_ratios = [0.1,1,10,100]
    ffactors = [my_ef_ratios]

    # selection method (only tournament is currently implemented)
    selection_method = 'tournament'

    # cross over (parenting) and mutation hyperparameters
    r_cross = 0.9
    r_mut = 0.1

    # convergence threshold for full function (value of RMSE E + RMSE F at which simulation is terminated" 
    convthr = 0.005 
    # fraction of ngenerations to start checking for convergence (convergence checks wont be performed very early)
    conv_check = 1.2

    #End optimization hyperparameters
    #---------------------------------------------------------------------------

    # set up generation 0
    best_eval = 9999999.9999
    conv_flag = False
    first_seeds = seedsi[:population_size+1]
    hp = HyperparameterStruct(ne,nf,eranges,ffactors)
    population = [hp.random_params(inputseed=first_seeds[ip]) for ip in range(population_size)]
    generation = 0

    best_evals = [best_eval]
    best = tuple(population[0])
    sim_seeds = seedsi[population_size:]
    np.random.seed(sim_seeds[generation])
    w_combo_delta = np.ones(len(gtks))
    # delta function to zero out force weights on structures without forces
    ef_rat_delta = np.array([1.0 if 'Volume' not in gti else 0.0 for gti in gtks])
    while generation <= ngenerations and best_eval > convthr and not conv_flag:
        scores = []
        # current generation
        for creature in population:
            creature_ew, creature_ffac = tuple(creature.reshape(2,ne).tolist())
            creature_ew = tuple(creature_ew)
            creature_ffac = tuple(creature_ffac)
            update_weights(creature_ew, creature_ffac, gtks = gtks, size_b = size_b, grouptype = grouptype)
            costi = fit_and_cost(snap,[etot_weight,ftot_weight])
            scores.append(costi)
            #NOTE to add another contribution to the cost function , you need to evaluate it in the loop
            # and add it to the fit_and_cost function
            # if this involves a lammps simulation, you will have to print potentials at the different steps
            # to run the lammps/pylammps simulation. To do so, the fitsnap output file name prefix should
            # be updated per step, then snap.write_output() should be called per step. This will likely increase
            # the optimization time.
        pt.single_print('generation,scores,popsize:',generation,len(scores),population_size)
        for i in range(population_size):
            if scores[i] < best_eval:
                best, best_eval = tuple(population[i]), scores[i]
        best_evals.append(best_eval)
        try:
            conv_flag = np.round(np.var(best_evals[int(ngenerations/conv_check)-int(ngenerations/10):]),14) == 0
        except IndexError:
            conv_flag = False
        printbest = tuple([tuple(ijk) for ijk in np.array(best).reshape(2,ne).tolist()])
        pt.single_print('generation:',generation, 'score:',scores[i])#generation, score
        print_final(printbest,train_sz=1.0,test_sz=0.0)
        slct = Selector(selection_style = selection_method)
        selected = [slct.selection(population, scores) for creature_idx in range(population_size)]
        del slct

        # new generation
        children = list()
        for ii in range(0, population_size, 2):
            # get selected parents in pairs
            p1, p2 = selected[ii], selected[ii+1]
            # crossover and mutation
            rndcross, rndmut = tuple(np.random.rand(2).tolist())
            if rndcross <= r_cross:
                cs = crossover(p1,p2,len(gtks),w_combo_delta,ef_rat_delta)
            else:
                cs = [p1,p2]
            for c in cs:
                # mutation
                if rndmut <= r_mut:
                    current_creature_ew, current_creature_ffac = tuple(c.reshape(2,ne))
                    current_creature_ew = tuple(current_creature_ew)
                    current_creature_ffac = tuple(current_creature_ffac)

                    mutated_creature_ew, mutated_creature_ffac = mutation(current_creature_ew,current_creature_ffac,my_w_ranges,my_ef_ratios,ng=len(gtks),w_combo_delta=w_combo_delta,ef_rat_delta=ef_rat_delta, apply_random=True,full_mutation=False)

                    c = np.append(mutated_creature_ew,mutated_creature_ffac)
                    # store for next generation
                children.append(c)
        generation += 1
        np.random.seed(sim_seeds[generation])
        population = children
    best_ew, best_ffac = tuple(np.array(best).reshape(2,ne).tolist())
    best_ew = tuple(creature_ew)
    best_ffac = tuple(creature_ffac)
    update_weights(best_ew, best_ffac, gtks = gtks, size_b = size_b, grouptype = grouptype)
    costi = fit_and_cost(snap,[etot_weight,ftot_weight])
    print_final(tuple([best_ew,best_ffac]))
    time2 = time.time()
    pt.single_print('Total optimization time,', time2 - time1)
    snap.write_output()

if optimization_style == 'simulated_annealing':
    sim_anneal()
elif optimization_style == 'genetic_algorithm':
    genetic_algorithm()
